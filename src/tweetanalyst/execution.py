"""Order execution for Polymarket — SELLS ONLY (PHASE 2 — built but NOT activated).

This builds the *capability* to place sell orders on positions, with the safest sensible design. It
is inert until you deliberately turn it on, and even then it will only ever SELL.

Security & safety model (enforced in code, not just convention):

  * **Disabled by default.** ``EXECUTION_ENABLED`` is False → every executor refuses to send.
  * **Dry-run is the default executor.** It validates and logs the order it *would* place, and
    returns without touching the network. You see exactly what would happen first.
  * **Sells only.** ``OrderIntent`` rejects any side other than SELL. There is no code path to buy.
  * **Explicit per-order confirmation.** A live send requires ``confirm=True`` passed at the call
    site (so a click/typed confirmation in the UI, never an automatic loop).
  * **Secrets live in the OS keychain**, never in the repo, never in plaintext env if avoidable. We
    use the ``keyring`` library (macOS Keychain / Windows Credential Locker / libsecret). The private
    key is read only at the moment of signing and never logged.
  * **Recommended: a dedicated trading wallet** funded with only what you're willing to trade — the
    private key's blast radius is then bounded to that wallet.

Why a private key at all: Polymarket's CLOB requires an EIP-712 *signature* of each order (L1), plus
API credentials (L2) that are themselves derived from the key. Orders cannot be placed without the
key signing them, so the security story is about *storing and using the key safely*, which is what
the keychain + dedicated-wallet + sells-only + dry-run model above is for.

Heavy deps (``py_clob_client``, ``keyring``) are imported lazily so this module imports fine before
they're installed; they're only needed once you actually enable live execution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("tweetanalyst.execution")

# ----- master switches (flip deliberately, never in code that runs unattended) ----- #
EXECUTION_ENABLED = False        # nothing is ever sent while this is False
ALLOW_BUY = False                # hard guard: there is no supported buy path; keep False

_KEYRING_SERVICE = "tweetanalyst.polymarket"
CLOB_HOST = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137


# --------------------------------------------------------------------------- #
# Order intent / result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OrderIntent:
    """A request to SELL part/all of a held position. Construction validates the sells-only rule."""
    market_slug: str
    bracket_label: str
    token_id: str            # the CLOB token (YES/NO outcome) being sold
    shares: float            # number of shares to sell (>0)
    limit_price: float       # minimum acceptable price per share, in (0, 1)
    side: str = "SELL"
    reason: str = ""         # human note (e.g. "edge flipped negative at τ=0.83")

    def __post_init__(self) -> None:
        if self.side != "SELL":
            raise ValueError(f"only SELL is supported (got {self.side!r}); buying is disabled")
        if not (self.shares > 0):
            raise ValueError("shares must be > 0")
        if not (0.0 < self.limit_price < 1.0):
            raise ValueError("limit_price must be in (0, 1)")


@dataclass
class ExecResult:
    intent: OrderIntent
    status: str              # "dry_run" | "submitted" | "rejected" | "error"
    detail: str = ""
    order_id: str | None = None


# --------------------------------------------------------------------------- #
# Executors
# --------------------------------------------------------------------------- #
class Executor:
    """Base: validates intents. Subclasses decide whether/how to actually send."""

    def _validate(self, intent: OrderIntent) -> None:
        if intent.side != "SELL" or ALLOW_BUY:
            raise ValueError("sells-only: refusing a non-SELL order")

    def submit(self, intent: OrderIntent, confirm: bool = False) -> ExecResult:  # pragma: no cover
        raise NotImplementedError


class DryRunExecutor(Executor):
    """Default. Logs the order it would place and returns — never touches the network."""

    def submit(self, intent: OrderIntent, confirm: bool = False) -> ExecResult:
        self._validate(intent)
        detail = (f"[DRY-RUN] SELL {intent.shares:.2f} shares of {intent.bracket_label} "
                  f"({intent.market_slug}) @ ≥ {intent.limit_price:.3f} — {intent.reason}")
        logger.info(detail)
        return ExecResult(intent=intent, status="dry_run", detail=detail)


class ClobExecutor(Executor):
    """Live Polymarket CLOB sell — DISABLED unless ``EXECUTION_ENABLED`` AND ``confirm=True``.

    Lazily builds a ``py_clob_client`` from the keychain-stored private key and posts a limit SELL.
    Any missing dependency / credential / switch yields a clean ``rejected``/``error`` result rather
    than an exception at import time."""

    def submit(self, intent: OrderIntent, confirm: bool = False) -> ExecResult:
        self._validate(intent)
        if not EXECUTION_ENABLED:
            return ExecResult(intent, "rejected", "execution disabled (EXECUTION_ENABLED is False)")
        if not confirm:
            return ExecResult(intent, "rejected", "explicit confirm=True required for a live order")
        try:
            from py_clob_client.client import ClobClient            # lazy
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import SELL
        except Exception as e:  # noqa: BLE001
            return ExecResult(intent, "error", f"py_clob_client not installed: {e}")
        key = load_private_key()
        if not key:
            return ExecResult(intent, "error", "no private key in keychain (see store_private_key)")
        try:
            client = ClobClient(CLOB_HOST, key=key, chain_id=POLYGON_CHAIN_ID)
            client.set_api_creds(client.create_or_derive_api_creds())  # L2 creds derived from the key
            order = client.create_order(OrderArgs(
                token_id=intent.token_id, price=intent.limit_price,
                size=intent.shares, side=SELL))
            resp = client.post_order(order, OrderType.GTC)
            oid = resp.get("orderID") if isinstance(resp, dict) else None
            return ExecResult(intent, "submitted", f"posted: {resp}", order_id=oid)
        except Exception as e:  # noqa: BLE001
            return ExecResult(intent, "error", f"order failed: {e}")


def get_executor(live: bool = False) -> Executor:
    """Return the dry-run executor (default) or the live CLOB one. Live still obeys all the guards."""
    return ClobExecutor() if (live and EXECUTION_ENABLED) else DryRunExecutor()


# --------------------------------------------------------------------------- #
# Secret storage (OS keychain via `keyring`; lazy)
# --------------------------------------------------------------------------- #
def store_private_key(private_key: str, account: str = "default") -> None:
    """Store the trading wallet's private key in the OS keychain (run once, interactively)."""
    import keyring  # lazy
    keyring.set_password(_KEYRING_SERVICE, account, private_key.strip())


def load_private_key(account: str = "default") -> str | None:
    try:
        import keyring  # lazy
        return keyring.get_password(_KEYRING_SERVICE, account)
    except Exception:  # noqa: BLE001
        return None


def delete_private_key(account: str = "default") -> None:
    import keyring  # lazy
    try:
        keyring.delete_password(_KEYRING_SERVICE, account)
    except Exception:  # noqa: BLE001
        pass
