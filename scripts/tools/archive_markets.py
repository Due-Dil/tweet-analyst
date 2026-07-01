"""One-shot / periodic backfill of every resolved Elon market into the local cache at 1-minute
fidelity, so backtests run fully offline and survive Polymarket's limited price-history retention.

    python archive_markets.py            # full backfill, all durations
    python archive_markets.py 2          # only 2-day markets
    python archive_markets.py status     # show what's already archived

Idempotent: markets already recorded are skipped (no re-download). Safe to run any time — e.g. weekly,
or after a market closes. The app also archives freshly-closed markets automatically on launch.
"""
import sys, time
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
sys.path.insert(0, "src")
import warnings
warnings.filterwarnings("ignore")
from tweetanalyst import archive as A


def _fmt(res: dict) -> str:
    return (f"{res['scanned']} marchés scannés · {len(res['new'])} nouveaux archivés "
            f"({res['points_added']:,} points de prix + {res.get('trades_added', 0):,} trades) · "
            f"{res['skipped']} déjà en stock")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "status":
        df = A.archive_status()
        print(f"=== {len(df)} marchés archivés localement ===")
        print(df.to_string(index=False) if len(df) else "(rien encore)")
        sys.exit(0)

    durations = (float(arg),) if arg else None
    t0 = time.time()
    print(f"Archivage 1-min des marchés résolus{' (' + arg + 'j)' if arg else ' (toutes durées)'}…",
          flush=True)

    def prog(i, n, slug):
        if i % 10 == 0 or i == n:
            print(f"  [{i}/{n}] {slug} ({time.time()-t0:.0f}s)", flush=True)

    res = A.archive_all(durations=durations, on_progress=prog)
    print(f"\n✅ {_fmt(res)}  —  {time.time()-t0:.0f}s")
    if res["new"]:
        print("Nouveaux marchés :", ", ".join(res["new"][:20]) + ("…" if len(res["new"]) > 20 else ""))
