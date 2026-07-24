"""Recalibrate the per-duration sharpening γ on the current regime, and persist it.

Run manually or on a schedule (e.g. weekly cron) to keep the model's calibration current as
Elon's posting regime drifts:

    python recalibrate.py            # refresh durations that are stale (>7 days)
    python recalibrate.py --force    # recalibrate all durations now
"""
import argparse
import datetime as dt
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
sys.path.insert(0, "src")

from tweetanalyst import calibration as C  # noqa: E402
from tweetanalyst import data as D  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--durations", default="2,3,7", help="comma-separated market durations in days")
    ap.add_argument("--max-age-days", type=float, default=7.0)
    ap.add_argument("--force", action="store_true", help="recalibrate even if fresh")
    ap.add_argument("--n-sims", type=int, default=3000)
    args = ap.parse_args()

    durations = [float(x) for x in args.durations.split(",")]
    now = dt.datetime.now(dt.timezone.utc)
    posts = D.ensure_history("elonmusk", days=130)  # incremental refresh first
    print(f"posts: {len(posts)} | anchor: {now.isoformat()}", flush=True)

    for dur in durations:
        if not args.force and not C.is_stale(dur, args.max_age_days):
            info = C.load_calibration().get(int(round(dur)))
            print(f"  {dur:.0f}d: fresh (γ={info['gamma']:.2f}, calibré {info['calibrated_at'][:10]}) — skip")
            continue
        print(f"  {dur:.0f}d: calibrating…", flush=True)
        # Real bracket width by market type: weekly ~20, short 2-3 day markets ~25.
        width = 20 if dur >= 6 else 25
        r = C.calibrate_gamma(posts, now, dur, n_sims=args.n_sims, bracket_width=width)
        C.store_calibration(dur, r)
        print(f"  {dur:.0f}d: γ={r['gamma']:.2f}  ({r['n_windows']} windows, "
              f"log-loss {r['ll_before']:.2f}->{r['ll_after']:.2f})", flush=True)

    print("done. Current table:", {d: i["gamma"] for d, i in C.load_calibration().items()})


if __name__ == "__main__":
    main()
