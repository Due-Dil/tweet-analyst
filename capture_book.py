"""Poll the LIVE order book (depth) for currently-open Elon markets, to enable a liquidity-accurate
autonomous backtest later (true fills / slippage, not mid±fixed-spread).

    python capture_book.py [interval_s=120] [duration_h=8]

Historical book depth is NOT available from the API, so we must capture it live as markets trade.
Appends one JSONL line per (token, poll) to data/orderbook/book_<UTCdate>.jsonl:
    {ts, slug, label, side, token, bids:[[price,size]...], asks:[[price,size]...], mid}
Both YES and NO tokens per bracket are captured (NO is quoted independently). Runs for ``duration_h``.
"""
import sys, time, json, datetime as dt
sys.path.insert(0, "src")
from pathlib import Path

import requests
from tweetanalyst import data as D

INTERVAL = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
DURATION_H = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
GAMMA, CLOB = "https://gamma-api.polymarket.com", "https://clob.polymarket.com"
S = requests.Session()


def active_tokens():
    """[(slug, label, side, token)] for every bracket of every currently-open Elon market."""
    now = dt.datetime.now(dt.timezone.utc)
    out = []
    for tw in D.get_trackings("elonmusk"):
        if not (tw.is_active and tw.market_link and tw.end > now):
            continue
        slug = D.slug_from_url(tw.market_link)
        try:
            ev = S.get(f"{GAMMA}/events", params={"slug": slug}, timeout=20).json()
        except Exception:  # noqa: BLE001
            continue
        if not ev:
            continue
        for m in ev[0].get("markets", []):
            lab = m.get("groupItemTitle") or ""
            toks, oc = m.get("clobTokenIds"), m.get("outcomes")
            toks = json.loads(toks) if isinstance(toks, str) else toks
            oc = json.loads(oc) if isinstance(oc, str) else oc
            if not toks:
                continue
            for i, tok in enumerate(toks):
                side = (oc[i].upper() if oc and i < len(oc) else f"OUT{i}")
                out.append((slug, lab, side, tok))
    return out


def fetch_book(token):
    try:
        b = S.get(f"{CLOB}/book", params={"token_id": token}, timeout=15).json()
        bids = [[float(x["price"]), float(x["size"])] for x in (b.get("bids") or [])]
        asks = [[float(x["price"]), float(x["size"])] for x in (b.get("asks") or [])]
        return bids, asks
    except Exception:  # noqa: BLE001
        return None, None


def main():
    out_dir = Path("backtest_data/orderbook")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"book_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d')}.jsonl"
    tokens = active_tokens()
    print(f"capture carnet: {len(tokens)} tokens sur {len({t[0] for t in tokens})} marchés ouverts "
          f"→ {path} (intervalle {INTERVAL:.0f}s, durée {DURATION_H:.0f}h)", flush=True)
    end = time.time() + DURATION_H * 3600
    cycle = 0
    with path.open("a", encoding="utf-8") as fh:
        while time.time() < end:
            ts = int(time.time())
            wrote = 0
            for (slug, lab, side, tok) in tokens:
                bids, asks = fetch_book(tok)
                if bids is None:
                    continue
                mid = ((bids[-1][0] + asks[-1][0]) / 2 if bids and asks else None)
                fh.write(json.dumps({"ts": ts, "slug": slug, "label": lab, "side": side,
                                     "token": tok, "bids": bids, "asks": asks, "mid": mid}) + "\n")
                wrote += 1
                time.sleep(0.08)            # gentle on the API
            fh.flush()
            cycle += 1
            print(f"[{dt.datetime.utcfromtimestamp(ts)}] cycle {cycle}: {wrote} carnets écrits",
                  flush=True)
            # refresh the token set hourly (new markets open / old close)
            if cycle % int(max(1, 3600 / INTERVAL)) == 0:
                tokens = active_tokens()
            slept = time.time() - ts
            if slept < INTERVAL:
                time.sleep(INTERVAL - slept)
    print("capture terminée", flush=True)


if __name__ == "__main__":
    main()
