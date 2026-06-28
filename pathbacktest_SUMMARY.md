# Path-backtest results — naive vs joint (correlation-corrected)

Full unified run, 138 resolved markets (63×7-day + 75×2-day), τ=2h, real CLOB prices, net of spread.
`arch-` duplicate excluded → 135–137 markets per variant. Raw CSVs: `pathbacktest_*_full_joint.csv`.
ROI = realized PnL / total cash staked. **To revisit later.**

## All durations
| Variant | ROI agg | ROI median/wk | σ/wk | win% wks | staked | pnl |
|---|---|---|---|---|---|---|
| V0_hold (naive buy&hold) | 32.8% | 7.4% | 0.91 | 61.5% | $72.9k | $23.9k |
| V1_rebalance (naive, **best naive**) | 51.9% | 28.9% | 0.74 | 90.4% | $178.6k | $92.6k |
| V2_profit_take (naive) | 47.9% | 26.6% | 0.72 | 88.9% | $188k | $90.2k |
| V3_reactive (naive) | 45.0% | 24.0% | 0.67 | 89.6% | $199.6k | $89.8k |
| V0_joint_hold (corrected buy&hold) | 27.6% | 0.1% | 1.40 | 50.4% | $22.9k | $6.3k |
| **V1_joint (corrected rebalance)** | **56.6%** | **32.3%** | 0.90 | 87.6% | **$80.2k** | $45.4k |

## Key reads
- **Correlation correction (joint Kelly) deploys ~½ the capital** ($80k vs $178k staked) yet earns a
  **higher per-dollar ROI** (56.6% vs 51.9%) and higher median (32.3% vs 28.9%). It does NOT lower the
  per-dollar return — the naive version was spraying capital on low-edge correlated NO bets that drag
  the average; joint concentrates on high-edge brackets + holds cash. Trade-off: higher variance
  (0.90 vs 0.74) and a few more losing weeks (87.6% vs 90.4% win) from concentration.
- **Rebalancing + cutting still dominates buy-and-hold under BOTH sizings**: V0_joint_hold (concentrate
  then never cut) is the worst (median ~0%, 50% win) — so "cut when model edge flips" matters MORE when
  positions are concentrated.
- **2-day ≈ 7-day**: V1_joint 2j median 25.9% / 7j 37.6%; same ranking on both. Short format holds up.

## Bottom line for sizing real money
Use **V1_joint** as the realistic basis: deploy ~half what naive suggests, expect a similar-or-better
per-dollar return but lumpier (concentrated) outcomes. Caveats still apply: γ mildly in-sample, thin
late-week fills, no live spread/slippage beyond the modeled haircut.
