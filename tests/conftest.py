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


class MockUser:
    """Mock Telethon User with configurable attributes."""

    def __init__(self, id=1, deleted=False, bot=False, scam=False, fake=False,
                 restricted=False, status=None, photo=True, username="testuser",
                 first_name="Test", last_name="User", premium=False,
                 emoji_status=None):
        self.id = id
        self.deleted = deleted
        self.bot = bot
        self.scam = scam
        self.fake = fake
        self.restricted = restricted
        self.status = status
        self.photo = "photo_placeholder" if photo else None
        self.username = username
        self.first_name = first_name
        self.last_name = last_name
        self.premium = premium
        self.emoji_status = emoji_status


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
