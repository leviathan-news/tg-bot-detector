"""
Admin log collector: discovers users from channel admin action log.

Uses GetAdminLogRequest to fetch the channel's admin action log, filtering to
join, leave, ban, and kick events. This gives a historical view of who has
subscribed to the channel (including users who may have since left or been
banned) and provides accurate join timestamps from the event dates.

Requires the calling account to have admin privileges on the channel. If the
account lacks admin access (ChatAdminRequired), the collector logs a warning
and returns whatever partial data it accumulated before the error.
"""

import asyncio
import sys

from .base import CollectorResult

from telethon.tl.functions.channels import GetAdminLogRequest
from telethon.tl.types import (
    ChannelAdminLogEventActionParticipantJoin,
    ChannelAdminLogEventActionParticipantLeave,
    ChannelAdminLogEventActionParticipantInvite,
    ChannelAdminLogEventActionParticipantToggleBan,
    ChannelAdminLogEventActionParticipantToggleAdmin,
)

try:
    # ChatAdminRequired lives in telethon.errors in most Telethon versions.
    from telethon.errors import ChatAdminRequiredError
except ImportError:
    # Fallback: define a dummy exception so the except clause still compiles
    # even if the import path changes in future Telethon versions.
    ChatAdminRequiredError = Exception  # type: ignore[misc,assignment]


# Event action types that are interesting for user discovery.
# Join events give us accurate join timestamps; leave/ban/kick events still
# tell us the account was once a member.
_JOIN_LEAVE_BAN_TYPES = (
    ChannelAdminLogEventActionParticipantJoin,
    ChannelAdminLogEventActionParticipantLeave,
    ChannelAdminLogEventActionParticipantInvite,
    ChannelAdminLogEventActionParticipantToggleBan,
    ChannelAdminLogEventActionParticipantToggleAdmin,
)


async def collect(
    client,
    channel,
    limit: int = 1000,
    delay: float = 1.0,
) -> CollectorResult:
    """Collect users from the channel's admin action log.

    Fetches join/leave/ban/kick events in batches of 100 until `limit` total
    events have been scanned or the log is exhausted. Users are indexed from
    the result.users list returned by each GetAdminLogRequest call. Join dates
    are extracted from the event timestamp (event.date).

    Args:
        client: Connected Telethon TelegramClient.
        channel: Resolved Telethon channel entity.
        limit: Maximum number of log events to scan. Default 1000.
        delay: Seconds to sleep between consecutive API calls. Default 1.0.

    Returns:
        CollectorResult with:
            source="admin_log"
            users: dict[int, User] — deduplicated users from log events
            join_dates: dict[int, datetime] — event.date for join events
            metadata["events_scanned"]: int — total events fetched

    Note:
        If ChatAdminRequiredError is raised (the calling account is not an
        admin), a warning is printed to stderr and the partial result
        accumulated so far is returned rather than raising.
    """
    users: dict = {}
    join_dates: dict = {}
    events_scanned = 0
    max_id = 0             # Pagination cursor (0 = start from newest event).
    batch_size = 100

    while events_scanned < limit:
        remaining = limit - events_scanned
        fetch_count = min(batch_size, remaining)

        try:
            result = await client(GetAdminLogRequest(
                channel=channel,
                q="",           # Empty query string = fetch all event types.
                max_id=max_id,
                min_id=0,
                limit=fetch_count,
            ))
        except ChatAdminRequiredError as e:
            # The calling account does not have the admin_log privilege.
            # Return whatever we have rather than crashing the whole pipeline.
            print(
                f"  admin_log collector: insufficient permissions — {e}",
                file=sys.stderr,
            )
            break
        except Exception as e:
            print(
                f"  admin_log collector: error fetching admin log — {e}",
                file=sys.stderr,
            )
            break

        # result.users contains all users referenced in this batch of events.
        for user in result.users:
            if user.id not in users:
                users[user.id] = user

        for event in result.events:
            events_scanned += 1

            # Extract join date from join events only. Leave/ban events tell
            # us the account existed but do not represent a subscription action.
            if isinstance(event.action, ChannelAdminLogEventActionParticipantJoin):
                # event.user_id is the subject of the event.
                if event.user_id not in join_dates:
                    join_dates[event.user_id] = event.date

        batch_len = len(result.events)

        # End of log: fewer events than requested means no more history.
        if batch_len < fetch_count:
            break

        # Advance pagination cursor to the oldest event ID in this batch.
        if result.events:
            max_id = result.events[-1].id

        await asyncio.sleep(delay)

    return CollectorResult(
        source="admin_log",
        users=users,
        join_dates=join_dates,
        metadata={"events_scanned": events_scanned},
    )
