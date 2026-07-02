import re

from modules.brain import Brain


class EmailDrafter:
    def __init__(self, config: dict):
        self.brain = Brain(config)
        self.response_name = config["personalize"].get("response_name", "")
        self.debug = True

    def classify_importance(self, email: dict) -> dict:
        """
        Classify email importance and detect special types.
        Returns dict with importance, type, suggested_action.
        """
        prompt = f"""Analyze this email and respond in JSON only:

            From: {email['from']}
            Subject: {email['subject']}
            Preview: {email['snippet']}

            Respond with ONLY this JSON:
            {{
              "importance": "high|medium|low",
              "type": "meeting_request|action_required|newsletter|notification|personal|other",
              "summary": "one sentence summary",
              "suggested_action": "reply|schedule_meeting|no_action",
              "meeting_date": "extracted date if meeting request, else null",
              "meeting_time": "extracted time if meeting request, else null",
              "meeting_duration": "extracted duration in minutes if mentioned, else 60"
            }}"""

        try:
            response = self.brain.query(prompt, model_key="orchestrator")
            import json
            import re
            # strip mark down if present
            clean = re.sub(r'```json|```', '', response).strip()
            return json.loads(clean)
        except Exception as e:
            print(f"[EmailDrafter] Classification error: {e}")
            return {
                "importance": "medium",
                "type": "other",
                "summary": email['snippet'][:100],
                "suggested_action": "no_action",
                "meeting_date": None,
                "meeting_time": None,
                "meeting_duration": 60
            }

        # ── Fast rule-based (no model) ────────────────────────────────────────

    def classify_importance_fast(self, email: dict) -> dict:
        """Fast rule-based classification — no LLM needed."""
        subject = email['subject'].lower()
        sender = email['from'].lower()
        snippet = email['snippet'].lower()
        combined = f"{subject} {snippet}"

        urgent_words = [
            "urgent", "asap", "immediately", "action required",
            "important", "critical", "deadline", "overdue",
            "invoice", "payment", "expires", "expiring",
            "security", "verify", "suspended", "locked"
        ]
        meeting_words = [
            "meeting", "schedule", "invite", "calendar",
            "zoom", "teams", "call", "interview", "appointment"
        ]
        promo_words = [
            "unsubscribe", "newsletter", "no-reply", "noreply",
            "marketing", "promotion", "offer", "deal", "sale",
            "linkedin", "twitter", "facebook", "notification"
        ]

        is_promo = any(w in combined for w in promo_words) or \
                   any(w in sender for w in ["noreply", "no-reply", "newsletter", "marketing"])
        is_meeting = any(w in combined for w in meeting_words)
        is_urgent = any(w in combined for w in urgent_words)

        if is_promo:
            importance = "low"
            email_type = "newsletter"
        elif is_urgent:
            importance = "high"
            email_type = "action_required"
        elif is_meeting:
            importance = "medium"
            email_type = "meeting_request"
        else:
            importance = "medium"
            email_type = "other"

        return {
            "importance": importance,
            "type": email_type,
            "summary": email['snippet'][:100],
            "suggested_action": "reply" if is_meeting or is_urgent else "no_action"
        }

    def summarize_inbox_fast(self, emails: list) -> str:
        """Fast rule-based inbox summary — no LLM."""
        if not emails:
            return f"Your inbox is clear, {self.response_name}."

        classified = [self.classify_importance_fast(e) for e in emails]
        high = [e for e, c in zip(emails, classified) if c['importance'] == 'high']
        meetings = [e for e, c in zip(emails, classified) if c['type'] == 'meeting_request']
        low = [e for e, c in zip(emails, classified) if c['importance'] == 'low']
        other = len(emails) - len(high) - len(low)

        parts = []
        if high:
            senders = ", ".join([e['from'].split('<')[0].strip() for e in high[:2]])
            parts.append(f"{len(high)} urgent from {senders}")
        if meetings:
            parts.append(f"{len(meetings)} meeting request{'s' if len(meetings) > 1 else ''}")
        if other > 0:
            parts.append(f"{other} other")
        if low:
            parts.append(f"{len(low)} newsletters")

        total = f"You have {len(emails)} unread emails — "
        return total + ", ".join(parts) + f", {self.response_name}."

        # ── Mistral-powered (only when needed) ────────────────────────────────

    def analyze_importance(self, emails: list) -> str:
        """Deep analysis — Mistral reviews emails for action items."""
        if not emails:
            return f"Nothing requiring attention, {self.response_name}."

        email_list = "\n".join([
            f"- From {e['from'].split('<')[0].strip()}: {e['subject']} — {e['snippet'][:100]}"
            for e in emails[:5]
        ])

        prompt = f"""Review these emails and identify what needs attention.
            Be concise — 2-3 sentences max. End with {self.response_name}'.
            Focus on action items, deadlines, meeting requests.
            Ignore newsletters and promotions.
        
            Emails:
            {email_list}
        
            Response:"""

        try:
            response = self.brain.query(
                prompt,
                model_key="orchestrator",
                num_ctx_override=2048,  # should this be increased?
                max_tokens_override=200  # should this be increased?
            )
            return self._clean(response)
        except Exception as e:
            print(f"[EmailDrafter] Analysis error: {e}")
            return self.summarize_inbox_fast(emails)

    def draft_reply(self, email: dict, instruction: str = None) -> str:
        """Draft a reply using Mistral — only called when user asks."""
        instruction_text = f"\nInstruction: {instruction}" if instruction else ""

        prompt = f"""Write a short professional email reply. 
            2-3 sentences. Plain text only. No subject line. No greeting. Just the body.{instruction_text}
        
            Original email:
            From: {email['from']}
            Subject: {email['subject']}
            Content: {email.get('body', email['snippet'])[:500]}
        
            Reply:"""

        try:
            response = self.brain.query(
                prompt,
                model_key="orchestrator",
                num_ctx_override=2048,
                max_tokens_override=300
            )
            return self._clean(response)
        except Exception as e:
            print(f"[EmailDrafter] Draft error: {e}")
            return "Thank you for your email. I'll get back to you shortly."

    def draft_new_email(self, to: str, subject: str, instruction: str) -> str:
        """Draft a new email from scratch."""
        prompt = f"""Write a short professional email.
            2-3 sentences. Plain text only. No subject line. No greeting. Just the body.
            To: {to}
            Subject: {subject}
            Instruction: {instruction}
        
            Email body:"""

        try:
            response = self.brain.query(
                prompt,
                model_key="orchestrator",
                num_ctx_override=2048,
                max_tokens_override=300
            )
            return self._clean(response)
        except Exception as e:
            print(f"[EmailDrafter] New email error: {e}")
            return instruction

        # ── Helpers ───────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        """Strip mark down from model output."""
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'#{1,6}\s?', '', text)
        text = re.sub(r'`+', '', text)
        text = re.sub(r'\n+', ' ', text)
        return text.strip()

    # overkill?
    # def draft_reply(self, email: dict, instruction: str = None) -> str:
    #     """
    #     Draft a reply to an email using Gemini.
    #     instruction: optional user instruction like "say yes" or "decline politely"
    #     """
    #     instruction_text = f"\nUser instruction: {instruction}" if instruction else ""
    #
    #     prompt = f"""Draft a professional but friendly email reply.
    #         Be concise — 2-3 sentences max unless the situation requires more.
    #         Do not include subject line or greeting — just the body text.
    #         Sign off as the user, not as an AI.{instruction_text}
    #
    #         Original email:
    #         From: {email['from']}
    #         Subject: {email['subject']}
    #         Content: {email.get('body', email['snippet'])}
    #
    #         Reply body only:"""
    #
    #     try:
    #         response = self.brain.query(prompt, model_key="gemini")
    #         # strip markdown
    #         import re
    #         response = re.sub(r'\*+', '', response)
    #         response = re.sub(r'#{1,6}\s?', '', response)
    #         return response.strip()
    #     except Exception as e:
    #         print(f"[EmailDrafter] Draft error: {e}")
    #         return "Thank you for your email. I'll get back to you shortly."

