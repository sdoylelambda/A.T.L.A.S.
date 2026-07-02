import asyncio
import datetime


async def handle_calendar_command(text: str, calendar, say, ears, stt, config) -> bool:
    """
    Handle calendar commands. Returns True if handled, False if not a calendar command.
    """
    if not calendar:
        return False

    response_name = config["personalize"].get("response_name", "")

    # ── Read intent ──────────────────────────────────────────────────────
    read_phrases = [
        "what's on my calendar", "what is on my calendar",
        "check my calendar", "show my calendar",
        "what do i have", "what's on my schedule"
    ]

    if any(phrase in text for phrase in read_phrases):
        await _handle_read(text, calendar, say)
        return True

    # ── Loose calendar words ──────────────────────────────────────────────
    calendar_words = [
        "today", "schedule", "events", "calendar", "this week",
        "upcoming", "next meeting", "next event", "next appointment"
    ]

    if any(word in text for word in calendar_words):
        if any(w in text for w in ["this week", "upcoming", "week"]):
            events = await asyncio.to_thread(calendar.get_upcoming_events, 7)
            await say(calendar.format_events_for_speech(events, multi_day=True), next_state="listening")
            return True

        if any(w in text for w in ["tomorrow", "tomorrow's"]):
            events = await asyncio.to_thread(calendar.get_tomorrows_events)
            await say(calendar.format_events_for_speech(events), next_state="listening")
            return True

        if any(w in text for w in ["next meeting", "next event", "what's next", "next appointment"]):
            event = await asyncio.to_thread(calendar.get_next_event)
            response = calendar.format_events_for_speech([event]) if event else f"Nothing coming up, {response_name}."
            await say(response, next_state="listening")
            return True

        if any(w in text for w in ["today", "schedule today", "my schedule"]):
            events = await asyncio.to_thread(calendar.get_todays_events)
            await say(calendar.format_events_for_speech(events), next_state="listening")
            return True

    # ── Create intent ─────────────────────────────────────────────────────
    create_triggers = ["add ", "schedule ", "create event", "new event",
                       "new appointment", "remind me", "set a reminder"]
    day_words = ["today", "tomorrow", "monday", "tuesday", "wednesday",
                 "thursday", "friday", "saturday", "sunday",
                 "tonight", "this evening", "next week"]

    if any(phrase in text for phrase in create_triggers) and \
       any(w in text for w in day_words):
        await _handle_create(text, calendar, say, ears, stt, config)
        return True

    return False


async def _handle_read(text: str, calendar, say) -> None:
    """Route read commands to correct time range."""
    if any(w in text for w in ["this week", "upcoming", "week"]):
        events = await asyncio.to_thread(calendar.get_upcoming_events, 7)
        await say(calendar.format_events_for_speech(events, multi_day=True), next_state="listening")
    elif any(w in text for w in ["tomorrow", "tomorrow's"]):
        events = await asyncio.to_thread(calendar.get_tomorrows_events)
        await say(calendar.format_events_for_speech(events), next_state="listening")
    else:
        events = await asyncio.to_thread(calendar.get_todays_events)
        await say(calendar.format_events_for_speech(events), next_state="listening")


async def _handle_create(text: str, calendar, say, ears, stt, config) -> None:
    """Handle event creation — parse directly or guided flow."""
    response_name = config["personalize"].get("response_name", "")
    parsed = calendar.parse_event_from_text(text)

    if parsed and parsed.get("time"):
        # have enough info — create directly
        await asyncio.to_thread(
            calendar.create_event,
            parsed["title"],
            parsed["date"],
            parsed.get("time"),
            parsed.get("duration", 60)
        )
        day_str = "today" if parsed["date"] == datetime.date.today().isoformat() else parsed["date"]
        await say(f"Done, {response_name}. {parsed['title']} added on {day_str} at {parsed.get('time')}.",
                  next_state="listening")

    elif parsed and not parsed.get("time"):
        # have title and date but no time
        await say(f"What time, {response_name}?")
        audio_bytes, dur = await ears.listen()
        time_text = stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""
        parsed = calendar.parse_event_from_text(f"{text} {time_text}")
        if parsed:
            await asyncio.to_thread(
                calendar.create_event,
                parsed["title"],
                parsed["date"],
                parsed.get("time"),
                parsed.get("duration", 60)
            )
            await say(f"Done, {response_name}. {parsed['title']} added.", next_state="listening")
        else:
            await say(f"Sorry {response_name}, I couldn't parse that. Please try again.", next_state="listening")

    else:
        # guided flow — ask one at a time
        await say(f"What's the title of the event, {response_name}?")
        audio_bytes, dur = await ears.listen()
        title = stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""

        await say(f"What day, {response_name}?")
        audio_bytes, dur = await ears.listen()
        day_text = stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""

        await say(f"What time, {response_name}?")
        audio_bytes, dur = await ears.listen()
        time_text = stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""

        parsed = calendar.parse_event_from_text(f"{title} {day_text} {time_text}")
        if parsed:
            await asyncio.to_thread(
                calendar.create_event,
                parsed["title"],
                parsed["date"],
                parsed.get("time"),
                parsed.get("duration", 60)
            )
            await say(f"Done, {response_name}. {parsed['title']} added.", next_state="listening")
        else:
            await say(f"Sorry {response_name}, I wasn't able to create that event.", next_state="listening")
