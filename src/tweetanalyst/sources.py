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
    """Low-latency feed of the most recent posts — provider-agnostic, DISABLED by default.

    Configure via env (kept local, never committed):
      * ``X_LIVE_BASE_URL`` — official X API v2 (``https://api.twitter.com/2``) OR a third-party
        provider's base URL (a cheap pay-as-you-go scraper API for our tiny volume).
      * ``X_LIVE_TOKEN``    — bearer/api token for that provider.
      * ``X_LIVE_USER_ID``  — the numeric account id (for v2 ``/users/:id/tweets``).

    Until ``LIVE_X_ENABLED`` is True AND those are set, ``fetch_recent`` returns ``[]`` so the rest of
    the system runs exactly as today. The concrete HTTP call + response mapping is intentionally a
    TODO: it depends on the chosen provider, and it MUST filter replies to match XTracker."""
    name = "x_live"

    def __init__(self) -> None:
        self.base_url = os.environ.get("X_LIVE_BASE_URL", "")
        self.token = os.environ.get("X_LIVE_TOKEN", "")
        self.user_id = os.environ.get("X_LIVE_USER_ID", "")

    @property
    def configured(self) -> bool:
        return bool(LIVE_X_ENABLED and self.base_url and self.token)

    def fetch_recent(self, handle: str, since: dt.datetime, until: dt.datetime) -> list[RawPost]:
        if not self.configured:
            return []
        # --- TODO (provider-specific, behind the flag) ---------------------------------------- #
        # import requests  (lazy)
        # GET {base_url}/users/{user_id}/tweets?start_time=...&max_results=...&tweet.fields=created_at,
        #     referenced_tweets  (auth: Bearer {token})
        # Map each item -> RawPost; DROP replies (referenced_tweets type == "replied_to" /
        # in_reply_to_user_id present) and keep main posts + quotes + reposts, so the count matches
        # XTracker's rule. Validate against XTracker overlap before trusting (see module docstring).
        raise NotImplementedError(
            "XLiveSource provider call not wired yet — choose official X API v2 or a pay-as-you-go "
            "third-party, set X_LIVE_* env vars, validate counts vs XTracker, then implement here.")


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
