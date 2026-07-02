import pytest
import datetime
from unittest.mock import MagicMock, patch
from modules.calendar_controller import CalendarModule


@pytest.fixture
def config():
    return {
        "integrations": {
            "google_calendar": {
                "enabled": True,
                "credentials_path": "~/.config/atlas/google_calendar_credentials.json",
                "token_path": "~/.config/atlas/google_calendar_token.json"
            }
        }
    }


@pytest.fixture
def calendar(config):
    cal = CalendarModule(config)
    cal.service = MagicMock()  # mock the Google API service
    return cal


# ─── parse_event_from_text ────────────────────────────────────────────────────

class TestParseEventFromText:

    def test_basic_event_tomorrow(self, calendar):
        result = calendar.parse_event_from_text("add dentist tomorrow at 2pm")
        tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
        assert result is not None
        assert result["date"] == tomorrow
        assert result["time"] == "14:00"
        assert "dentist" in result["title"].lower()

    def test_basic_event_today(self, calendar):
        result = calendar.parse_event_from_text("schedule a meeting today at 10am")
        assert result is not None
        assert result["date"] == datetime.date.today().isoformat()
        assert result["time"] == "10:00"

    def test_day_name(self, calendar):
        result = calendar.parse_event_from_text("add dentist friday at 2pm")
        assert result is not None
        assert result["time"] == "14:00"
        assert "dentist" in result["title"].lower()

    def test_time_with_period(self, calendar):
        result = calendar.parse_event_from_text("schedule a massage friday at 9.30am")
        assert result is not None
        assert result["time"] == "09:30"

    def test_time_with_colon(self, calendar):
        result = calendar.parse_event_from_text("add meeting tomorrow at 9:30am")
        assert result is not None
        assert result["time"] == "09:30"

    def test_24hr_time(self, calendar):
        result = calendar.parse_event_from_text("add standup tomorrow at 14:00")
        assert result is not None
        assert result["time"] == "14:00"

    def test_duration_hours(self, calendar):
        result = calendar.parse_event_from_text("add dentist friday at 2pm for 2 hours")
        assert result is not None
        assert result["duration"] == 120

    def test_duration_minutes(self, calendar):
        result = calendar.parse_event_from_text("add standup tomorrow at 9am for 30 minutes")
        assert result is not None
        assert result["duration"] == 30

    def test_default_duration(self, calendar):
        result = calendar.parse_event_from_text("add dentist tomorrow at 2pm")
        assert result is not None
        assert result["duration"] == 60

    def test_title_strips_leading_article(self, calendar):
        result = calendar.parse_event_from_text("add a doctors appointment tomorrow at 2pm")
        assert result is not None
        assert not result["title"].lower().startswith("a ")

    def test_title_strips_an(self, calendar):
        result = calendar.parse_event_from_text("schedule an interview tomorrow at 10am")
        assert result is not None
        assert not result["title"].lower().startswith("an ")

    def test_missing_date_returns_none(self, calendar):
        result = calendar.parse_event_from_text("add a meeting at 2pm")
        assert result is None

    def test_missing_title_returns_none(self, calendar):
        result = calendar.parse_event_from_text("add tomorrow at 2pm")
        assert result is None

    def test_pm_conversion(self, calendar):
        result = calendar.parse_event_from_text("add lunch tomorrow at 12pm")
        assert result is not None
        assert result["time"] == "12:00"

    def test_midnight(self, calendar):
        result = calendar.parse_event_from_text("add reminder tomorrow at 12am")
        assert result is not None
        assert result["time"] == "00:00"


# ─── format_events_for_speech ─────────────────────────────────────────────────

class TestFormatEventsForSpeech:

    def test_empty_events(self, calendar):
        result = calendar.format_events_for_speech([])
        assert "nothing" in result.lower()

    def test_single_event(self, calendar):
        events = [{"summary": "Dentist", "start": {"dateTime": "2026-03-17T14:00:00-04:00"}}]
        result = calendar.format_events_for_speech(events)
        assert "dentist" in result.lower()
        assert "2:00" in result
        assert "sir" in result.lower()

    def test_two_events(self, calendar):
        events = [
            {"summary": "Dentist", "start": {"dateTime": "2026-03-17T14:00:00-04:00"}},
            {"summary": "Massage", "start": {"dateTime": "2026-03-17T16:00:00-04:00"}},
        ]
        result = calendar.format_events_for_speech(events)
        assert "dentist" in result.lower()
        assert "massage" in result.lower()
        assert "and" in result.lower()

    def test_three_events(self, calendar):
        events = [
            {"summary": "Standup", "start": {"dateTime": "2026-03-17T09:00:00-04:00"}},
            {"summary": "Lunch", "start": {"dateTime": "2026-03-17T12:00:00-04:00"}},
            {"summary": "Dentist", "start": {"dateTime": "2026-03-17T14:00:00-04:00"}},
        ]
        result = calendar.format_events_for_speech(events)
        assert "standup" in result.lower()
        assert "lunch" in result.lower()
        assert "dentist" in result.lower()

    def test_all_day_event(self, calendar):
        events = [{"summary": "Holiday", "start": {"date": "2026-03-17"}}]
        result = calendar.format_events_for_speech(events)
        assert "all day" in result.lower()

    def test_multi_day_groups_by_day(self, calendar):
        today = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=1)
        events = [
            {"summary": "Standup", "start": {"dateTime": f"{today}T09:00:00-04:00"}},
            {"summary": "Dentist", "start": {"dateTime": f"{tomorrow}T14:00:00-04:00"}},
        ]
        result = calendar.format_events_for_speech(events, multi_day=True)
        assert "today" in result.lower()
        assert "tomorrow" in result.lower()
        assert "standup" in result.lower()
        assert "dentist" in result.lower()

    def test_multi_day_day_name_for_future(self, calendar):
        today = datetime.date.today()
        future = today + datetime.timedelta(days=3)
        events = [
            {"summary": "Meeting", "start": {"dateTime": f"{future}T10:00:00-04:00"}},
        ]
        result = calendar.format_events_for_speech(events, multi_day=True)
        assert "on" in result.lower()


# ─── create_event ─────────────────────────────────────────────────────────────

class TestCreateEvent:

    def test_create_event_with_time(self, calendar):
        calendar.service.events.return_value.insert.return_value.execute.return_value = {
            "id": "abc123",
            "summary": "Dentist"
        }
        result = calendar.create_event("Dentist", "2026-03-20", "14:00", 60)
        assert result["summary"] == "Dentist"
        calendar.service.events().insert.assert_called_once()

    def test_create_all_day_event(self, calendar):
        calendar.service.events.return_value.insert.return_value.execute.return_value = {
            "id": "abc123",
            "summary": "Holiday"
        }
        result = calendar.create_event("Holiday", "2026-03-20")
        assert result is not None
        call_args = calendar.service.events().insert.call_args
        body = call_args[1]["body"]
        assert "date" in body["start"]


# ─── get events ───────────────────────────────────────────────────────────────

class TestGetEvents:

    def test_get_todays_events_returns_list(self, calendar):
        calendar.service.events.return_value.list.return_value.execute.return_value = {
            "items": [{"summary": "Standup", "start": {"dateTime": "2026-03-17T09:00:00-04:00"}}]
        }
        result = calendar.get_todays_events()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_todays_events_empty(self, calendar):
        calendar.service.events.return_value.list.return_value.execute.return_value = {"items": []}
        result = calendar.get_todays_events()
        assert result == []

    def test_get_tomorrows_events_returns_list(self, calendar):
        calendar.service.events.return_value.list.return_value.execute.return_value = {
            "items": [{"summary": "Dentist", "start": {"dateTime": "2026-03-18T14:00:00-04:00"}}]
        }
        result = calendar.get_tomorrows_events()
        assert isinstance(result, list)

    def test_get_upcoming_events_returns_list(self, calendar):
        calendar.service.events.return_value.list.return_value.execute.return_value = {
            "items": [{"summary": "Meeting", "start": {"dateTime": "2026-03-19T10:00:00-04:00"}}]
        }
        result = calendar.get_upcoming_events(7)
        assert isinstance(result, list)

    def test_get_next_event_returns_dict(self, calendar):
        calendar.service.events.return_value.list.return_value.execute.return_value = {
            "items": [{"summary": "Dentist", "start": {"dateTime": "2026-03-18T14:00:00-04:00"}}]
        }
        result = calendar.get_next_event()
        assert result is not None
        assert result["summary"] == "Dentist"

    def test_get_next_event_none_when_empty(self, calendar):
        calendar.service.events.return_value.list.return_value.execute.return_value = {"items": []}
        result = calendar.get_next_event()
        assert result is None
