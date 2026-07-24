# Phase 2 — Live X feed + automatic SELL execution

Everything is **built but inert**. Activation is a few deliberate steps; nothing trades or calls X
until you flip the flags.

## What's in place
- **Low-latency X feed** (`sources.py`): cuts XTracker's ~5-min lag. XTracker stays the *resolution
  source of truth*; the live feed only adds the **fresh tail** (posts newer than XTracker's latest),
  so the count the market resolves on is never altered. Wired (inert) into `pipeline.run_forecast`.
- **Auto-sell engine** (`execution.py`): turns the strategy's **Sortir** (exit) and **Alléger** (trim
  to target) signals into **SELL** orders. Buys are never executed. Dry-run by default.

## Cost (X API, pay-per-use since Feb 2026)
$0.005 per post read, **deduplicated within a 24h UTC window** → polling frequency is free; you only
pay per *unique* post. Incremental fetch ⇒ ~**$4–5/mo** to ingest every Elon tweet all month, or
**~$1–2/mo** if you only run the final 24–48h of each market. One-time user-id lookup ~$0.01.

## Activation steps
1. **Install the extra deps** (base app doesn't need them):
   ```
   pip install -r requirements-live.txt
   ```
2. **Store credentials** in the OS keychain (never committed):
   ```
   python setup_credentials.py
   ```
   - X API **bearer token** (from the X developer console, pay-per-use).
   - Polymarket **private key** — use a **dedicated low-balance trading wallet**.
3. **Validate the X feed against XTracker** on a live week *before* trusting it: enable it, compare
   counts in the overlap region (they must match XTracker, which excludes replies).
4. **Turn things on** (independently, only when ready):
   - Live feed:    set `sources.LIVE_X_ENABLED = True`
   - Sell exec:    set `execution.EXECUTION_ENABLED = True`
5. **Run sells** — start in dry-run to preview, then go live with explicit confirmation:
   ```python
   from tweetanalyst import execution as X
   X.run_autosell(live=False)                 # preview (no orders sent)
   X.run_autosell(live=True, confirm=True)     # live SELLs (requires EXECUTION_ENABLED)
   ```

## Safety model (enforced in code)
- **Sells only** — `OrderIntent` rejects any non-SELL; there is no buy path.
- **Disabled by default** — both flags are `False`; `get_executor` returns the dry-run executor.
- **Explicit per-call confirmation** — live orders require `confirm=True`.
- **Secrets in the OS keychain** — read only at use time, never logged or committed.
- **Triggered by you** — `run_autosell` is meant to be called from a button / explicit live tick, not
  a hidden background loop.
