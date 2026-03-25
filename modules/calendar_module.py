import os
import datetime
import re
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/gmail.modify'
]


class CalendarModule:
    def __init__(self, config: dict):
        self.debug = True
        self.config = config.get("integrations", {}).get("google_calendar", {})
        self.creds_path = os.path.expanduser(self.config.get("credentials_path", "~/.config/atlas/google_calendar_credentials.json"))
        self.token_path = os.path.expanduser(self.config.get("token_path", "~/.config/atlas/google_calendar_token.json"))
        self.service = None

    def authenticate(self):
        """Authenticate and build the Calendar service."""
        creds = None

        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.creds_path, SCOPES)
                creds = flow.run_local_server(port=0)

            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())

        self.service = build('calendar', 'v3', credentials=creds)
        print("[Calendar] Authenticated successfully.")

    def _ensure_authenticated(self):
        if not self.service:
            self.authenticate()

    def get_todays_events(self) -> list:
        self._ensure_authenticated()
        # use local time not UTC
        now = datetime.datetime.now()
        start = datetime.datetime(now.year, now.month, now.day, 0, 0, 0).astimezone().isoformat()
        end = datetime.datetime(now.year, now.month, now.day, 23, 59, 59).astimezone().isoformat()

        if self.debug:
            print(f"[Calendar] today querying timeMin={start} timeMax={end}")

        events_result = self.service.events().list(
            calendarId='primary',
            timeMin=start,
            timeMax=end,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        items = events_result.get('items', [])
        if self.debug:
            print(f"[Calendar] found {len(items)} events")
            for item in items:
                print(f"[Calendar] event: {item.get('summary')} start={item.get('start')}")

        return items

    def get_tomorrows_events(self) -> list:
        """Get all events for tomorrow."""
        self._ensure_authenticated()
        tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
        start = datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0).astimezone().isoformat()
        end = datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day, 23, 59, 59).astimezone().isoformat()

        if self.debug:
            print(f"[Calendar] tomorrow querying timeMin={start} timeMax={end}")

        events_result = self.service.events().list(
            calendarId='primary',
            timeMin=start,
            timeMax=end,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        items = events_result.get('items', [])
        if self.debug:
            print(f"[Calendar] found {len(items)} events")
            for item in items:
                print(f"[Calendar] event: {item.get('summary')} start={item.get('start')}")

        return items

    def get_next_event(self) -> dict | None:
        """Get the next upcoming event."""
        self._ensure_authenticated()
        now = datetime.datetime.now().astimezone()
        now_str = now.isoformat()

        if self.debug:
            print(f"[Calendar] next event querying timeMin={now_str}")

        events_result = self.service.events().list(
            calendarId='primary',
            timeMin=now_str,
            maxResults=1,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        items = events_result.get('items', [])
        if self.debug:
            print(f"[Calendar] found {len(items)} events")
            for item in items:
                print(f"[Calendar] event: {item.get('summary')} start={item.get('start')}")

        return items[0] if items else None

    def get_upcoming_events(self, days: int = 7) -> list:
        self._ensure_authenticated()
        now = datetime.datetime.now()
        end = now + datetime.timedelta(days=days)
        start = now.astimezone().isoformat()
        end_str = datetime.datetime(end.year, end.month, end.day, 23, 59, 59).astimezone().isoformat()

        if self.debug:
            print(f"[Calendar] querying timeMin={start} timeMax={end_str}")

        events_result = self.service.events().list(
            calendarId='primary',
            timeMin=start,
            timeMax=end_str,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        items = events_result.get('items', [])
        if self.debug:
            print(f"[Calendar] found {len(items)} events")
            for item in items:
                print(f"[Calendar] event: {item.get('summary')} start={item.get('start')}")

        return items

    def parse_event_from_text(self, text: str) -> dict | None:
        """
        Try to extract event details from natural language.
        Returns dict with title, date, time, duration or None if can't parse.
        """
        today = datetime.date.today()
        result = {}

        # --- extract date ---
        if "today" in text:
            result["date"] = today.isoformat()
        elif "tomorrow" in text:
            result["date"] = (today + datetime.timedelta(days=1)).isoformat()
        else:
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            for i, day in enumerate(days):
                if day in text:
                    current_weekday = today.weekday()
                    days_ahead = (i - current_weekday) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    result["date"] = (today + datetime.timedelta(days=days_ahead)).isoformat()
                    break

            # --- extract time ---
        time_match = re.search(
            r'\b(\d{1,2})(?:[:\.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b',
            text, re.IGNORECASE
        )
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            meridiem = time_match.group(3).lower().replace('.', '')
            if meridiem == 'pm' and hour != 12:
                hour += 12
            elif meridiem == 'am' and hour == 12:
                hour = 0
            result["time"] = f"{hour:02d}:{minute:02d}"
        else:
            time_match_24 = re.search(r'\b(\d{2}):(\d{2})\b', text)
            if time_match_24:
                result["time"] = f"{time_match_24.group(1)}:{time_match_24.group(2)}"

        # --- extract duration ---
        duration_match = re.search(r'for\s+(\d+)\s*(hour|hr|minute|min)', text, re.IGNORECASE)
        if duration_match:
            amount = int(duration_match.group(1))
            unit = duration_match.group(2).lower()
            result["duration"] = amount * 60 if "hour" in unit or "hr" in unit else amount
        else:
            result["duration"] = 60

        # --- extract title — strip everything that isn't the title ---
        title_text = text

        # strip scheduling words
        for word in ["add", "schedule", "create", "new", "event", "appointment",
                     "today", "tomorrow", "monday", "tuesday",
                     "wednesday", "thursday", "friday", "saturday", "sunday",
                     "for an hour", "for a hour", "for 30 minutes",
                     " a ", " an ", " at ", " on "]:
            title_text = title_text.replace(word, "")

        # strip time match from title
        if time_match:
            title_text = title_text[:time_match.start()] + title_text[time_match.end():]
        elif time_match_24:
            title_text = title_text[:time_match_24.start()] + title_text[time_match_24.end():]

        # strip duration match
        if duration_match:
            title_text = title_text[:duration_match.start()] + title_text[duration_match.end():]

        # strip remaining time patterns
        title_text = re.sub(r'\b\d{1,2}[:.]\d{2}\s*(am|pm)?\b', '', title_text, flags=re.IGNORECASE)
        title_text = re.sub(r'\b\d{1,2}\s*(am|pm|a\.m\.|p\.m\.)\b', '', title_text, flags=re.IGNORECASE)

        # clean up
        title_text = re.sub(r'\s+', ' ', title_text).strip(" ,.-")
        title_text = re.sub(r'^(a|an)\s+', '', title_text, flags=re.IGNORECASE).strip()

        if title_text and len(title_text) > 2:
            result["title"] = title_text

        if result.get("title") and result.get("date"):
            return result

        return None

    async def guided_create_event(self, say, listen) -> dict | None:
        """Ask user for event details one at a time."""
        await say("What's the title of the event, sir?")
        audio, duration = await listen()
        if not audio:
            return None
        title = audio  # raw transcription

        await say("What day, sir?")
        audio, duration = await listen()
        if not audio:
            return None
        day_text = audio

        await say("What time, sir?")
        audio, duration = await listen()
        if not audio:
            return None
        time_text = audio

        # parse the collected pieces together
        combined = f"{title} {day_text} {time_text}"
        return self.parse_event_from_text(combined)

    def create_event(self, title: str, date: str, time: str = None, duration_minutes: int = 60) -> dict:
        """Create a new calendar event."""
        self._ensure_authenticated()

        if time:
            start_dt = datetime.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
            end_dt = start_dt + datetime.timedelta(minutes=duration_minutes)
            event = {
                'summary': title,
                'start': {'dateTime': start_dt.isoformat(), 'timeZone': 'America/New_York'},
                'end': {'dateTime': end_dt.isoformat(), 'timeZone': 'America/New_York'},
            }
        else:
            event = {
                'summary': title,
                'start': {'date': date},
                'end': {'date': date},
            }

        return self.service.events().insert(calendarId='primary', body=event).execute()

    def format_events_for_speech(self, events: list, multi_day: bool = False) -> str:
        """Convert events list to natural speech string."""
        if not events:
            return "Nothing on the calendar, sir."

        if not multi_day:
            # original single day format
            parts = []
            for event in events:
                title = event.get('summary', 'Untitled event')
                start = event.get('start', {})
                if 'dateTime' in start:
                    dt = datetime.datetime.fromisoformat(start['dateTime'].replace('Z', '+00:00'))
                    local_dt = dt.astimezone()
                    time_str = local_dt.strftime("%I:%M %p").lstrip("0")
                    parts.append(f"{title} at {time_str}")
                else:
                    parts.append(f"{title} all day")

            if len(parts) == 1:
                return parts[0] + ", sir."
            elif len(parts) == 2:
                return f"{parts[0]} and {parts[1]}, sir."
            else:
                return ", ".join(parts[:-1]) + f", and {parts[-1]}, sir."

        else:
            # multi-day format — group by day
            today = datetime.datetime.now().date()
            tomorrow = today + datetime.timedelta(days=1)

            # group events by date
            from collections import defaultdict
            by_day = defaultdict(list)
            for event in events:
                start = event.get('start', {})
                if 'dateTime' in start:
                    dt = datetime.datetime.fromisoformat(start['dateTime'].replace('Z', '+00:00')).astimezone()
                    day = dt.date()
                    time_str = dt.strftime("%I:%M %p").lstrip("0")
                    by_day[day].append(f"{event.get('summary', 'Untitled')} at {time_str}")
                else:
                    day = datetime.date.fromisoformat(start.get('date', str(today)))
                    by_day[day].append(f"{event.get('summary', 'Untitled')} all day")

            # build speech string day by day
            sentences = []
            for day in sorted(by_day.keys()):
                events_for_day = by_day[day]

                if day == today:
                    day_label = "Today"
                elif day == tomorrow:
                    day_label = "Tomorrow"
                else:
                    day_label = f"On {day.strftime('%A the %d').replace(' 0', ' ')}"

                if len(events_for_day) == 1:
                    day_str = f"{day_label} you have {events_for_day[0]}."
                elif len(events_for_day) == 2:
                    day_str = f"{day_label} you have {events_for_day[0]} and {events_for_day[1]}."
                else:
                    day_str = f"{day_label} you have {', '.join(events_for_day[:-1])}, and {events_for_day[-1]}."

                sentences.append(day_str)

            return " ".join(sentences) + " Sir."
