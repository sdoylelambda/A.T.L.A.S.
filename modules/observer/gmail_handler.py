import asyncio


async def handle_gmail_command(text: str, gmail, drafter, calendar, say, ears, stt) -> bool:
    """
    Handle Gmail voice commands. Returns True if handled.
    """
    if not gmail:
        return False

    # ── Read unread ───────────────────────────────────────────────────────
    # fast — no model
    if any(phrase in text for phrase in [
        "check my emails", "read my emails", "any new emails", "what's in my inbox"
        "check my inbox", "any unread emails", "what emails do i have", "check my mail"
    ]):
        emails = await asyncio.to_thread(gmail.get_unread, 5)
        summary = drafter.summarize_inbox_fast(emails)  # ← fast, no model
        await say(summary, next_state="listening")
        return True

    # smart — Mistral kicks in
    if any(phrase in text for phrase in [
        "anything important", "anything urgent", "analyze my inbox",
        "what should i focus on", "any action items", "summarize my inbox"
    ]):
        emails = await asyncio.to_thread(gmail.get_unread, 10)
        analysis = await asyncio.to_thread(drafter.analyze_importance, emails)
        await say(analysis, next_state="listening")
        return True

    # With Mistral --- is there a use case?
    # if any(phrase in text for phrase in [
    #     "read my emails", "check my emails", "any new emails",
    #     "check my inbox", "any unread emails", "what emails do i have"
    # ]):
    #     emails = await asyncio.to_thread(gmail.get_unread, 5)
    #     if not emails:
    #         await say("No unread emails, sir.", next_state="listening")
    #         return True
    #     summary = await asyncio.to_thread(drafter.summarize_inbox, emails)
    #     await say(summary, next_state="listening")
    #     return True

    # ── Emails from specific person ───────────────────────────────────────
    if "emails from" in text or "email from" in text:
        # extract name after "from"
        parts = text.split("from")
        name = parts[-1].strip() if len(parts) > 1 else ""
        if name:
            emails = await asyncio.to_thread(gmail.get_emails_from, name, 3)
            response = gmail.format_emails_for_speech(emails)
            await say(response, next_state="listening")
        else:
            await say("Who should I search for, sir?", next_state="listening")
        return True

    # ── Read specific email ───────────────────────────────────────────────
    if any(phrase in text for phrase in [
        "read that email", "read the email", "read it",
        "what does it say", "read the first email"
    ]):
        emails = await asyncio.to_thread(gmail.get_unread, 1)
        if not emails:
            await say("No unread emails to read, sir.", next_state="listening")
            return True
        email = emails[0]
        body = await asyncio.to_thread(gmail.get_email_body, email['id'])
        sender = email['from'].split('<')[0].strip()
        await say(f"Email from {sender}. Subject: {email['subject']}. {body[:500]}", next_state="listening")
        return True

    # ── Smart inbox summary ───────────────────────────────────────────────
    if any(phrase in text for phrase in [
        "summarize my inbox", "inbox summary", "what's important",
        "any important emails", "anything urgent"
    ]):
        emails = await asyncio.to_thread(gmail.get_unread, 10)
        if not emails:
            await say("Inbox is clear, sir.", next_state="listening")
            return True

        # classify each email
        important = []
        for email in emails:
            classification = await asyncio.to_thread(drafter.classify_importance, email)
            email['classification'] = classification
            if classification.get('importance') == 'high':
                important.append(email)

        if important:
            parts = [f"{e['from'].split('<')[0].strip()} about {e['subject']}" for e in important[:3]]
            response = f"You have {len(important)} important email{'s' if len(important) > 1 else ''}. "
            response += ", and ".join(parts) + ", sir."
        else:
            response = f"You have {len(emails)} unread emails, none flagged as urgent, sir."

        await say(response, next_state="listening")
        return True

    # ── Draft reply ───────────────────────────────────────────────────────
    if any(phrase in text for phrase in [
        "reply to", "respond to", "draft a reply", "write a reply"
    ]):
        emails = await asyncio.to_thread(gmail.get_unread, 1)
        if not emails:
            await say("No emails to reply to, sir.", next_state="listening")
            return True

        email = emails[0]
        sender = email['from'].split('<')[0].strip()

        # ask for instruction
        await say(f"Drafting reply to {sender}. What should I say, sir?")
        audio_bytes, dur = await ears.listen()
        instruction = stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""

        draft_body = await asyncio.to_thread(drafter.draft_reply, email, instruction)
        draft_id = await asyncio.to_thread(gmail.reply_to_email, email['id'], draft_body)

        await say(f"Draft saved. Here's what I wrote: {draft_body[:200]}")
        await say("Shall I send it, sir?")

        audio_bytes, dur = await ears.listen()
        response = stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""

        confirmed = any(w in response for w in ["yes", "yeah", "send", "do it", "sure"])
        if confirmed:
            await asyncio.to_thread(gmail.send_draft, draft_id)
            await say("Sent, sir.", next_state="listening")
        else:
            await say("Draft saved in Gmail for your review, sir.", next_state="listening")
        return True

    # ── Send new email ────────────────────────────────────────────────────
    if any(phrase in text for phrase in [
        "send an email", "send email", "email to", "write an email"
    ]):
        # extract recipient name
        for phrase in ["send an email to", "send email to", "email to", "write an email to"]:
            if phrase in text:
                name = text.split(phrase)[-1].strip()
                break
        else:
            name = ""

        if not name:
            await say("Who should I send it to, sir?")
            audio_bytes, dur = await ears.listen()
            name = stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""

        # find email address
        to_email = await asyncio.to_thread(gmail.find_contact_email, name)
        if not to_email:
            await say(f"I couldn't find an email address for {name}, sir.", next_state="listening")
            return True

        # get subject
        await say("What's the subject, sir?")
        audio_bytes, dur = await ears.listen()
        subject = stt.transcribe(audio_bytes, dur).strip() if audio_bytes else "No subject"

        # get message
        await say("What should it say, sir?")
        audio_bytes, dur = await ears.listen()
        instruction = stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""

        # draft it
        draft_body = await asyncio.to_thread(
            drafter.draft_reply,
            {'from': '', 'subject': subject, 'snippet': '', 'body': instruction},
            instruction
        )

        # save draft
        draft_id = await asyncio.to_thread(gmail.save_draft, to_email, subject, draft_body)

        await say(f"Draft ready. Here's what I wrote: {draft_body[:200]}")
        await say("Shall I send it, sir?")

        audio_bytes, dur = await ears.listen()
        response = stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""

        confirmed = any(w in response for w in ["yes", "yeah", "send", "do it", "sure"])
        if confirmed:
            await asyncio.to_thread(gmail.send_draft, draft_id)
            await say("Sent, sir.", next_state="listening")
        else:
            await say("Draft saved in Gmail for your review, sir.", next_state="listening")
        return True

    # ── Meeting request detection ─────────────────────────────────────────
    if any(phrase in text for phrase in [
        "any meeting requests", "check for meetings",
        "any invites", "meeting invitations"
    ]):
        emails = await asyncio.to_thread(gmail.get_unread, 10)
        meeting_emails = []
        for email in emails:
            classification = await asyncio.to_thread(drafter.classify_importance, email)
            if classification.get('type') == 'meeting_request':
                email['classification'] = classification
                meeting_emails.append(email)

        if not meeting_emails:
            await say("No meeting requests in your inbox, sir.", next_state="listening")
            return True

        for email in meeting_emails[:2]:
            c = email['classification']
            sender = email['from'].split('<')[0].strip()
            await say(
                f"Meeting request from {sender}. {c.get('summary', email['subject'])}. "
                f"Shall I draft a reply and add it to your calendar, sir?"
            )

            audio_bytes, dur = await ears.listen()
            response = stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""
            confirmed = any(w in response for w in ["yes", "yeah", "sure", "do it"])

            if confirmed and calendar and c.get('meeting_date'):
                # add to calendar
                await asyncio.to_thread(
                    calendar.create_event,
                    email['subject'],
                    c['meeting_date'],
                    c.get('meeting_time'),
                    int(c.get('meeting_duration', 60))
                )
                # draft acceptance
                draft_body = await asyncio.to_thread(
                    drafter.draft_reply, email,
                    "accept the meeting and confirm the time"
                )
                draft_id = await asyncio.to_thread(
                    gmail.reply_to_email, email['id'], draft_body
                )
                await say(
                    f"Added to calendar and draft reply saved. "
                    f"Here's the reply: {draft_body[:150]}. Shall I send it, sir?"
                )

                audio_bytes, dur = await ears.listen()
                send_response = stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""
                if any(w in send_response for w in ["yes", "yeah", "send", "sure"]):
                    await asyncio.to_thread(gmail.send_draft, draft_id)
                    await say("Sent and calendar updated, sir.", next_state="listening")
                else:
                    await say("Draft saved in Gmail for your review, sir.", next_state="listening")

        return True

    return False
