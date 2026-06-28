"""Read a wallet's open Polymarket positions on the Elon-tweet markets and compare them to the model.

Positions come from Polymarket's public Data API (`data-api.polymarket.com/positions?user=<address>`),
read-only by wallet address — no key/signature needed. For each open Elon-tweet position we look up
the current model probability for that bracket, compute the edge on the side actually held (YES/NO),
and flag whether the bet is still aligned with the model or worth reconsidering.
"""
from __future__ import annotations

import datetime as dt
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

from . import pipeline as P

DATA_API = "https://data-api.polymarket.com"
ELON_PREFIX = "elon-musk-of-tweets"

# Local, git-ignored store for the user's own wallet address (so they don't retype it). Never committed.
_WALLET_FILE = Path(__file__).resolve().parents[2] / "data" / "wallet.txt"


def load_wallet() -> str:
    try:
        return _WALLET_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def save_wallet(address: str) -> None:
    _WALLET_FILE.parent.mkdir(parents=True, exist_ok=True)
    _WALLET_FILE.write_text(address.strip(), encoding="utf-8")


def fetch_positions(address: str, limit: int = 500) -> list[dict]:
    """All current positions for ``address`` (any market). Raises on network error."""
    r = requests.get(f"{DATA_API}/positions", params={"user": address.strip(), "limit": limit},
                     timeout=30)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def _bracket_bounds_from_title(title: str) -> tuple[float, float] | None:
    """'...post 200-219 tweets...' -> (200, 219); handles '<N', 'N-M', and 'N+'."""
    m = re.search(r"post\s+<\s*([0-9]+)\s+tweets", title)
    if m:
        return 0.0, float(m.group(1)) - 1
    m = re.search(r"post\s+([0-9]+)\s*-\s*([0-9]+)\s+tweets", title)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"post\s+([0-9]+)\+\s+tweets", title)
    if m:
        return float(m.group(1)), float("inf")
    return None


def open_elon_positions(positions: list[dict], now: dt.datetime | None = None) -> list[dict]:
    """Keep only Elon-tweet positions in markets that haven't closed yet (still forecastable)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    out = []
    for p in positions:
        if not str(p.get("eventSlug", "")).startswith(ELON_PREFIX):
            continue
        if float(p.get("size", 0)) <= 0:
            continue
        end = p.get("endDate")
        if end:
            try:
                if pd.Timestamp(end).tz_convert("UTC").to_pydatetime() <= now:
                    continue  # market already closed
            except Exception:  # noqa: BLE001
                pass
        out.append(p)
    return out


def analyze(address: str, n_sims: int = 12000, now: dt.datetime | None = None) -> dict:
    """Return {positions: [...], summary: {...}} comparing the wallet's open Elon bets to the model.

    One model forecast per distinct market (positions are grouped), then each bracket position is
    matched to its model probability and the edge of the held side.
    """
    raw = open_elon_positions(fetch_positions(address), now=now)
    by_market: dict[str, list[dict]] = defaultdict(list)
    for p in raw:
        by_market[p["eventSlug"]].append(p)

    rows: list[dict] = []
    for slug, ps in by_market.items():
        try:
            run = P.run_forecast(slug, now=now, n_sims=n_sims, refresh=False)
        except Exception as e:  # noqa: BLE001  (skip markets the model can't price)
            for p in ps:
                rows.append(_row(p, None, None, error=str(e)))
            continue
        for p in ps:
            bounds = _bracket_bounds_from_title(p.get("title", ""))
            tbl = None
            if bounds is not None:
                mid = bounds[0] if bounds[1] == float("inf") else (bounds[0] + bounds[1]) / 2
                tbl = next((t for t in run.table
                            if t["low"] <= mid <= (t["high"] if t["high"] != float("inf") else 1e9)),
                           None)
            rows.append(_row(p, tbl, run.market.title))

    df_rows = rows
    n = len(df_rows)
    at_risk = sum(r["valeur_actuelle"] for r in df_rows)
    cost = sum(r["mise"] for r in df_rows)
    pnl = sum(r["pnl"] for r in df_rows)
    misaligned = [r for r in df_rows if r["statut"] == "⚠️ Réajuster"]
    summary = {
        "n_positions": n,
        "n_markets": len(by_market),
        "valeur_actuelle": at_risk,
        "mise_totale": cost,
        "pnl_total": pnl,
        "gain_max_total": sum(r["gain_max"] for r in df_rows),
        "rendement_max_pct": (sum(r["gain_max"] for r in df_rows) / cost) if cost > 0 else None,
        "n_misaligned": len(misaligned),
        "exposition_a_revoir": sum(r["valeur_actuelle"] for r in misaligned),
    }
    return {"positions": df_rows, "summary": summary}


def _row(p: dict, tbl: dict | None, market_title: str | None, error: str | None = None) -> dict:
    size = float(p.get("size", 0))
    avg = float(p.get("avgPrice", 0))
    cur = float(p.get("curPrice", 0))
    side = (p.get("outcome") or "").upper()  # "YES" / "NO"
    mise = size * avg
    valeur = float(p.get("currentValue", size * cur))
    pnl = float(p.get("cashPnl", valeur - mise))
    # Each share pays $1 if the held side wins -> max payout = size; max profit = size - cost.
    gain_max = size - mise
    rendement_max = (gain_max / mise) if mise > 0 else None  # (1-entry)/entry

    model_prob = edge = status = None
    p_in = tbl["model_prob"] if tbl else None
    if tbl is not None:
        if side == "YES":
            model_prob = p_in
            yes_price = tbl["yes_price"] if tbl["yes_price"] is not None else cur
            edge = model_prob - yes_price
        else:  # NO
            model_prob = 1.0 - p_in
            no_price = tbl["no_price"] if tbl["no_price"] is not None else (1.0 - cur)
            edge = model_prob - no_price
        if edge is None:
            status = "—"
        elif edge > 0.03:
            status = "✅ Aligné"
        elif edge < -0.03:
            status = "⚠️ Réajuster"
        else:
            status = "≈ Neutre"
    else:
        status = "❓ Non évalué"

    return {
        "marché": market_title or p.get("eventSlug", ""),
        "tranche": _label_from_title(p.get("title", "")),
        "côté": side,
        "parts": size,
        "prix_entrée": avg,
        "prix_marché": cur,
        "mise": mise,
        "valeur_actuelle": valeur,
        "pnl": pnl,
        "gain_max": gain_max,
        "rendement_max": rendement_max,
        "proba_modèle_côté": model_prob,
        "edge_côté": edge,
        "statut": status if not error else "❓ Non évalué",
    }


def _label_from_title(title: str) -> str:
    b = _bracket_bounds_from_title(title)
    if b is None:
        return title
    lo, hi = b
    if lo == 0:
        return f"<{int(hi) + 1}"
    return f"{int(lo)}+" if hi == float("inf") else f"{int(lo)}-{int(hi)}"
