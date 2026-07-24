"""Full performance history on the Elon-tweet markets — realized + latent (unrealized).

Polymarket's ``/positions`` only shows *open* positions, so a market that resolved and was redeemed
disappears from it. To get the COMPLETE history we reconstruct cash flows from ``/activity`` (every
TRADE buy/sell + REDEEM at resolution + SPLIT/MERGE), and add the current mark of any still-open
position. Per market:

    net_cash      = (sells + redeems + merges) − (buys + splits)
    current_value = Σ currentValue of open positions in that market (0 if fully closed)
    total_pnl     = net_cash + current_value
    latent (unrealized) = Σ cashPnl of open positions   (Polymarket's own unrealized number)
    realized      = total_pnl − latent                   (profit locked on sold/redeemed shares)

For a fully-closed market current_value = latent = 0, so realized = net_cash. Read-only by wallet
address (public Data API, no key)."""
from __future__ import annotations

from collections import defaultdict

import requests

from . import positions as POS

DATA_API = "https://data-api.polymarket.com"
ELON_PREFIX = "elon-musk-of-tweets"


def _fetch_activity(address: str, max_pages: int = 30) -> list[dict]:
    out: list[dict] = []
    off = 0
    for _ in range(max_pages):
        try:
            d = requests.get(f"{DATA_API}/activity", params={"user": address.strip(), "limit": 500,
                                                             "offset": off}, timeout=30).json()
        except Exception:  # noqa: BLE001
            break
        if not isinstance(d, list) or not d:
            break
        out.extend(d)
        off += len(d)
        if len(d) < 500:
            break
    return out


def _label(slug: str) -> str:
    """'elon-musk-of-tweets-june-19-june-26' -> 'june 19 → june 26'."""
    s = slug.replace(ELON_PREFIX + "-", "").replace("-", " ")
    parts = s.split()
    # join into 'month day → month day' when it looks like two month-day pairs
    return s if len(parts) < 4 else f"{parts[0]} {parts[1]} → {parts[2]} {parts[3]}"


def performance_history(address: str) -> dict:
    """Return {rows: [...per market...], totals: {...}} of realized + latent P&L on Elon markets."""
    if not address:
        return {"rows": [], "totals": {}}
    acts = _fetch_activity(address)
    positions = POS.fetch_positions(address)

    flows: dict[str, dict] = defaultdict(
        lambda: {"spent": 0.0, "received": 0.0, "n_trades": 0, "first_ts": 10 ** 11, "last_ts": 0})
    for a in acts:
        slug = str(a.get("eventSlug", ""))
        if not slug.startswith(ELON_PREFIX):
            continue
        typ, side = a.get("type"), a.get("side")
        usdc = float(a.get("usdcSize", 0) or 0)
        ts = int(a.get("timestamp", 0) or 0)
        f = flows[slug]
        if (typ == "TRADE" and side == "BUY") or typ == "SPLIT":
            f["spent"] += usdc
        elif (typ == "TRADE" and side == "SELL") or typ in ("REDEEM", "MERGE", "REWARD"):
            f["received"] += usdc
        if typ == "TRADE":
            f["n_trades"] += 1
        f["first_ts"] = min(f["first_ts"], ts)
        f["last_ts"] = max(f["last_ts"], ts)

    openm: dict[str, dict] = defaultdict(lambda: {"current_value": 0.0, "cashPnl": 0.0})
    for p in positions:
        slug = str(p.get("eventSlug", ""))
        if not slug.startswith(ELON_PREFIX):
            continue
        openm[slug]["current_value"] += float(p.get("currentValue", 0) or 0)
        openm[slug]["cashPnl"] += float(p.get("cashPnl", 0) or 0)

    rows = []
    for slug, f in flows.items():
        cv = openm[slug]["current_value"]
        latent = openm[slug]["cashPnl"]
        net_cash = f["received"] - f["spent"]
        total = net_cash + cv
        rows.append({
            "slug": slug, "label": _label(slug), "invested": f["spent"], "received": f["received"],
            "current_value": cv, "realized": total - latent, "latent": latent, "total": total,
            "roi": (total / f["spent"]) if f["spent"] > 0 else None,
            "is_open": cv > 1e-6, "n_trades": f["n_trades"], "last_ts": f["last_ts"],
        })
    rows.sort(key=lambda r: -r["last_ts"])

    closed = [r for r in rows if not r["is_open"]]
    inv = sum(r["invested"] for r in rows)
    tot = {
        "invested": inv, "realized": sum(r["realized"] for r in rows),
        "latent": sum(r["latent"] for r in rows), "total": sum(r["total"] for r in rows),
        "current_value": sum(r["current_value"] for r in rows),
        "roi": (sum(r["total"] for r in rows) / inv) if inv > 0 else None,
        "n_markets": len(rows), "n_open": sum(1 for r in rows if r["is_open"]), "n_closed": len(closed),
        "win_rate_closed": (sum(1 for r in closed if r["realized"] > 0) / len(closed)) if closed else None,
    }
    return {"rows": rows, "totals": tot}
