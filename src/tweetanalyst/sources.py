"""Pluggable post-source layer (PHASE 2 — built but NOT activated).

Today the model is fed by XTracker (``data.load_posts``), which lags X by ~5 min. Near a bracket
boundary that lag can cost us an early readjustment. This module lets us blend a **low-latency X
feed** on top of XTracker WITHOUT changing how the model consumes posts — and, crucially, without
ever letting the fast feed corrupt the number the market actually resolves on.

Design principles (read before wiring this live):

  * **XTracker is the resolution source of truth.** The market settles on XTracker's count (main-feed
    posts + quotes + reposts, replies excluded). So the fast feed is used ONLY to extend the *fresh
    tail* — posts newer than XTracker's latest — to react sooner. The settled region is always 100%
    XTracker. When XTracker catches up, its rows replace the provisional fast-feed tail.
  * **Same counting rules.** Any fast source MUST apply XTracker's inclusion rules (exclude replies)
    or it will drift from the resolution count. Reply filtering lives in each source.
  * **Pluggable + cheap.** ``XLiveSource`` is provider-agnostic (official X API v2 *or* a cheap
    pay-as-you-go third-party): point it at a base URL + auth via config. Disabled by default.

Nothing here is activated: ``LIVE_X_ENABLED`` is False, ``XLiveSource`` returns ``[]`` until
configured, and the existing ``data.load_posts`` path is untouched.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from . import data as D

# Master switch for the live blend. Leave False until a provider is configured AND validated against
# XTracker on a live week (counts must match in the overlap region before trusting the fast tail).
LIVE_X_ENABLED = False

# X API v2 (pay-per-use since Feb 2026). Default to the official host; overridable for a third-party.
X_API_BASE = os.environ.get("X_LIVE_BASE_URL", "https://api.x.com/2")
_X_KEYRING_SERVICE = "tweetanalyst.x"   # OS keychain entry for the bearer token
_USER_ID_CACHE: dict[str, str] = {}     # handle -> numeric user id (resolved once)


def store_x_token(bearer_token: str, account: str = "bearer") -> None:
    """Store the X API bearer token in the OS keychain (run once, e.g. via setup_credentials.py)."""
    import keyring  # lazy
    keyring.set_password(_X_KEYRING_SERVICE, account, bearer_token.strip())


def load_x_token(account: str = "bearer") -> str:
    """Bearer token from the OS keychain, falling back to the X_LIVE_TOKEN env var."""
    try:
        import keyring  # lazy
        tok = keyring.get_password(_X_KEYRING_SERVICE, account)
        if tok:
            return tok
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("X_LIVE_TOKEN", "")


@dataclass(frozen=True)
class RawPost:
    """One post, in the same shape the cache/model already use."""
    platform_id: str
    created_at: dt.datetime  # UTC-aware
    content: str
    is_repost: bool

    @property
    def is_reply(self) -> bool:  # convenience; sources set is_repost, replies are dropped upstream
        return False


@runtime_checkable
class PostSource(Protocol):
    name: str

    def fetch_recent(self, handle: str, since: dt.datetime, until: dt.datetime) -> list[RawPost]:
        """Posts in [since, until). Must already exclude replies (XTracker's counting rule)."""
        ...


# --------------------------------------------------------------------------- #
# Canonical source: XTracker (the market's resolution source)
# --------------------------------------------------------------------------- #
class XTrackerSource:
    """Wraps the existing XTracker fetch. This is the source of truth — never overridden by a fast
    feed in the settled region."""
    name = "xtracker"

    def fetch_recent(self, handle: str, since: dt.datetime, until: dt.datetime) -> list[RawPost]:
        rows = D._fetch_posts_remote(handle, since, until)  # already excludes replies (XTracker side)
        out = []
        for r in rows:
            out.append(RawPost(
                platform_id=r["platform_id"],
                created_at=pd.to_datetime(r["created_at"], utc=True).to_pydatetime(),
                content=r.get("content", "") or "",
                is_repost=bool(r.get("is_repost", 0)),
            ))
        return out


# --------------------------------------------------------------------------- #
# Low-latency source: X live feed (DISABLED — stub until a provider is chosen)
# --------------------------------------------------------------------------- #
class XLiveSource:
    """Low-latency feed of the most recent posts via the X API v2 (pay-per-use) — DISABLED by default.

    Reads ``GET /2/users/:id/tweets`` with ``exclude=replies`` (so we never even pay to read replies)
    and keeps main posts + quotes + reposts, matching XTracker's counting rule. Incremental by
    ``start_time`` (the caller passes the canonical latest timestamp), so cost ≈ new posts × $0.005.

    Auth: bearer token from the OS keychain (``store_x_token``) or the ``X_LIVE_TOKEN`` env var. The
    numeric user id is taken from ``X_LIVE_USER_ID`` or resolved once from the handle (cached). Until
    ``LIVE_X_ENABLED`` is True and a token is present, ``fetch_recent`` returns ``[]`` so the rest of
    the system runs exactly as today.
    """
    name = "x_live"

    def __init__(self) -> None:
        self.base_url = X_API_BASE
        self.token = load_x_token()
        self.user_id = os.environ.get("X_LIVE_USER_ID", "")

    @property
    def configured(self) -> bool:
        return bool(LIVE_X_ENABLED and self.base_url and self.token)

    def _resolve_user_id(self, handle: str) -> str:
        if self.user_id:
            return self.user_id
        if handle in _USER_ID_CACHE:
            return _USER_ID_CACHE[handle]
        import requests  # lazy
        r = requests.get(f"{self.base_url}/users/by/username/{handle}",
                         headers={"Authorization": f"Bearer {self.token}"}, timeout=20)
        r.raise_for_status()
        uid = str(r.json()["data"]["id"])
        _USER_ID_CACHE[handle] = uid
        return uid

    def fetch_recent(self, handle: str, since: dt.datetime, until: dt.datetime) -> list[RawPost]:
        if not self.configured:
            return []
        import requests  # lazy
        uid = self._resolve_user_id(handle)
        headers = {"Authorization": f"Bearer {self.token}"}
        params = {
            "max_results": 100,
            "exclude": "replies",                      # don't pay to read replies (XTracker excludes them)
            "tweet.fields": "created_at,referenced_tweets",
            "start_time": _rfc3339(since),
            "end_time": _rfc3339(until),
        }
        out: list[RawPost] = []
        for _ in range(5):  # paginate a few pages max (the fresh tail is tiny)
            r = requests.get(f"{self.base_url}/users/{uid}/tweets", headers=headers,
                             params=params, timeout=20)
            r.raise_for_status()
            body = r.json()
            for t in body.get("data", []) or []:
                refs = t.get("referenced_tweets") or []
                types = {x.get("type") for x in refs}
                if "replied_to" in types:            # belt-and-suspenders (exclude=replies already drops)
                    continue
                out.append(RawPost(
                    platform_id=str(t["id"]),
                    created_at=pd.to_datetime(t["created_at"], utc=True).to_pydatetime(),
                    content=t.get("text", "") or "",
                    is_repost=("retweeted" in types),
                ))
            nxt = body.get("meta", {}).get("next_token")
            if not nxt:
                break
            params["pagination_token"] = nxt
        return out


# --------------------------------------------------------------------------- #
# Safe blend: XTracker truth + fast tail
# --------------------------------------------------------------------------- #
def blend(canonical: list[RawPost], live: list[RawPost]) -> tuple[list[RawPost], dict]:
    """Merge a fast feed onto the canonical XTracker posts WITHOUT touching the settled region.

    Only live posts strictly **newer** than the canonical latest timestamp are added (the fresh
    tail). Everything up to XTracker's latest stays 100% XTracker. De-dup is by ``platform_id``.
    Returns ``(merged_sorted, freshness)`` where freshness reports each source's latest timestamp and
    the lag the fast feed is buying us."""
    by_id = {p.platform_id: p for p in canonical}
    canon_latest = max((p.created_at for p in canonical), default=None)
    added = 0
    for p in live:
        if p.platform_id in by_id:
            continue
        if canon_latest is not None and p.created_at <= canon_latest:
            continue  # never override/duplicate the settled region — XTracker owns it
        by_id[p.platform_id] = p
        added += 1
    merged = sorted(by_id.values(), key=lambda p: p.created_at)
    live_latest = max((p.created_at for p in live), default=None)
    lag = ((live_latest - canon_latest).total_seconds()
           if (live_latest and canon_latest and live_latest > canon_latest) else 0.0)
    freshness = {"canonical_latest": canon_latest, "live_latest": live_latest,
                 "fast_tail_added": added, "lag_seconds_recovered": lag}
    return merged, freshness


def posts_to_frame(posts: list[RawPost]) -> pd.DataFrame:
    """Same columns/dtypes as ``data.load_posts`` so the model consumes a blend identically."""
    df = pd.DataFrame([{"created_at": p.created_at, "content": p.content,
                        "is_repost": p.is_repost} for p in posts])
    if df.empty:
        return pd.DataFrame(columns=["created_at", "content", "is_repost"])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["is_repost"] = df["is_repost"].astype(bool)
    return df.sort_values("created_at").reset_index(drop=True)


def _rfc3339(d: dt.datetime) -> str:
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def augment_with_live(posts_df: pd.DataFrame, handle: str, now: dt.datetime,
                      lookback_minutes: float = 30.0) -> pd.DataFrame:
    """Append the fresh X tail to an XTracker DataFrame — INERT unless ``LIVE_X_ENABLED``.

    Only posts strictly newer than the canonical latest are added (XTracker owns the settled region),
    so the count the market resolves on is never altered. Safe to call unconditionally from the
    pipeline: returns ``posts_df`` unchanged when the live feed is disabled or empty."""
    if not LIVE_X_ENABLED:
        return posts_df
    try:
        canon_latest = (pd.Timestamp(posts_df["created_at"].max()).to_pydatetime()
                        if len(posts_df) else now - dt.timedelta(days=1))
        since = max(canon_latest, now - dt.timedelta(minutes=lookback_minutes))
        live = XLiveSource().fetch_recent(handle, since=since, until=now)
        fresh = [p for p in live if p.created_at > canon_latest]
        if not fresh:
            return posts_df
        add = posts_to_frame(fresh)
        return (pd.concat([posts_df, add], ignore_index=True)
                .sort_values("created_at").reset_index(drop=True))
    except Exception:  # noqa: BLE001  (never let a live hiccup break the forecast)
        return posts_df
