"""
Mock Telethon User fixtures for testing without a real Telegram connection.

Provides a `make_user` factory that creates objects with the same attribute
interface as Telethon's User type.
"""

import pytest
from datetime import datetime, timezone, timedelta


class MockUserStatus:
    """Base class for mock status types."""
    pass


class UserStatusEmpty(MockUserStatus):
    """User has never been seen online."""
    pass


class UserStatusOnline(MockUserStatus):
    """User is currently online."""
    pass


class UserStatusRecently(MockUserStatus):
    """User was online recently."""
    pass


class UserStatusLastWeek(MockUserStatus):
    """User was online within the last week."""
    pass


class UserStatusLastMonth(MockUserStatus):
    """User was online within the last month."""
    pass


class UserStatusOffline(MockUserStatus):
    """User was last seen at a specific time."""
    def __init__(self, was_online):
        self.was_online = was_online


class MockProfilePhoto:
    """Mock Telethon UserProfilePhoto with configurable metadata.

    Mirrors the real UserProfilePhoto fields available without downloading:
      - photo_id: unique identifier for the profile photo
      - dc_id: data center where the photo is stored (1-5)
      - has_video: whether the profile has an animated video avatar
      - stripped_thumb: tiny inline JPEG thumbnail bytes (typically 100-200 bytes)
    """

    def __init__(self, photo_id=123456, dc_id=2, has_video=False,
                 stripped_thumb=None):
        self.photo_id = photo_id
        self.dc_id = dc_id
        self.has_video = has_video
        self.stripped_thumb = stripped_thumb


class MockUser:
    """Mock Telethon User with configurable attributes."""

    def __init__(self, id=1, deleted=False, bot=False, scam=False, fake=False,
                 restricted=False, status=None, photo=True, username="testuser",
                 first_name="Test", last_name="User", premium=False,
                 emoji_status=None, photo_dc_id=None, photo_has_video=False,
                 photo_id=None, photo_stripped_thumb=None,
                 color=None, profile_color=None, stories_max_id=None,
                 stories_unavailable=False, contact_require_premium=False,
                 usernames=None, verified=False, lang_code=None,
                 send_paid_messages_stars=None):
        self.id = id
        self.deleted = deleted
        self.bot = bot
        self.scam = scam
        self.fake = fake
        self.restricted = restricted
        self.status = status
        self.username = username
        self.first_name = first_name
        self.last_name = last_name
        self.premium = premium
        self.emoji_status = emoji_status

        # Extended profile fields (MTProto layer 160+)
        self.color = color
        self.profile_color = profile_color
        self.stories_max_id = stories_max_id
        self.stories_unavailable = stories_unavailable
        self.contact_require_premium = contact_require_premium
        self.usernames = usernames
        self.verified = verified
        self.lang_code = lang_code
        self.send_paid_messages_stars = send_paid_messages_stars

        # Build photo attribute: either a MockProfilePhoto with metadata,
        # a simple placeholder string (backward compat), or None.
        if photo and (photo_dc_id is not None or photo_has_video or
                      photo_id is not None or photo_stripped_thumb is not None):
            self.photo = MockProfilePhoto(
                photo_id=photo_id or 123456,
                dc_id=photo_dc_id or 2,
                has_video=photo_has_video,
                stripped_thumb=photo_stripped_thumb,
            )
        elif photo:
            self.photo = "photo_placeholder"
        else:
            self.photo = None


@pytest.fixture
def make_user():
    """Factory fixture for creating mock Telethon User objects."""
    _counter = [0]

    def _make(**kwargs):
        _counter[0] += 1
        kwargs.setdefault("id", _counter[0])
        return MockUser(**kwargs)

    return _make


@pytest.fixture
def clean_user(make_user):
    """A typical clean (human-looking) user."""
    return make_user(
        first_name="Alice",
        last_name="Smith",
        username="alice_smith",
        photo=True,
        status=UserStatusRecently(),
    )


@pytest.fixture
def bot_user(make_user):
    """A typical bot-looking user."""
    return make_user(
        first_name="A",
        last_name=None,
        username=None,
        photo=False,
        status=None,
    )


@pytest.fixture
def deleted_user(make_user):
    """A deleted account."""
    return make_user(
        deleted=True,
        first_name=None,
        last_name=None,
        username=None,
        photo=False,
        status=None,
    )


@pytest.fixture
def premium_user(make_user):
    """A Telegram Premium user."""
    return make_user(
        first_name="Bob",
        last_name="Jones",
        username="bob_premium",
        photo=True,
        premium=True,
        emoji_status="star",
        status=UserStatusOnline(),
    )
