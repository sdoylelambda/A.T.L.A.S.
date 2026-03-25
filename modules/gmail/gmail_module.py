import os
import base64
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/gmail.modify'
]


class GmailModule:
    def __init__(self, config: dict):
        self.config = config.get("integrations", {}).get("gmail", {})
        self.creds_path = os.path.expanduser(
            config.get("integrations", {}).get("google_calendar", {}).get(
                "credentials_path", "~/.config/atlas/google_calendar_credentials.json"
            )
        )
        self.token_path = os.path.expanduser(
            config.get("integrations", {}).get("google_calendar", {}).get(
                "token_path", "~/.config/atlas/google_calendar_token.json"
            )
        )
        self.service = None
        self.debug = True

    def authenticate(self):
        """Authenticate and build Gmail service — reuses Calendar token."""
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
        self.service = build('gmail', 'v1', credentials=creds)
        print("[Gmail] Authenticated successfully.")

    def _ensure_authenticated(self):
        if not self.service:
            self.authenticate()

    # ── Read ─────────────────────────────────────────────────────────────

    def get_unread(self, max_results: int = 5) -> list:
        """Get unread emails."""
        self._ensure_authenticated()
        results = self.service.users().messages().list(
            userId='me',
            labelIds=['INBOX', 'UNREAD'],
            maxResults=max_results
        ).execute()
        messages = results.get('messages', [])
        return [self._parse_email(m['id']) for m in messages]

    def get_emails_from(self, sender: str, max_results: int = 5) -> list:
        """Search emails from a specific sender."""
        self._ensure_authenticated()
        results = self.service.users().messages().list(
            userId='me',
            q=f"from:{sender}",
            maxResults=max_results
        ).execute()
        messages = results.get('messages', [])
        return [self._parse_email(m['id']) for m in messages]

    def search_emails(self, query: str, max_results: int = 5) -> list:
        """Search emails by query."""
        self._ensure_authenticated()
        results = self.service.users().messages().list(
            userId='me',
            q=query,
            maxResults=max_results
        ).execute()
        messages = results.get('messages', [])
        return [self._parse_email(m['id']) for m in messages]

    def get_email_body(self, email_id: str) -> str:
        """Get full email body."""
        self._ensure_authenticated()
        email = self._parse_email(email_id, full=True)
        return email.get('body', 'No body found.')

    def _parse_email(self, email_id: str, full: bool = False) -> dict:
        """Parse email into clean dict."""
        msg = self.service.users().messages().get(
            userId='me',
            id=email_id,
            format='full' if full else 'metadata',
            metadataHeaders=['From', 'To', 'Subject', 'Date']
        ).execute()

        headers = {h['name']: h['value'] for h in msg['payload'].get('headers', [])}
        result = {
            'id': email_id,
            'from': headers.get('From', ''),
            'to': headers.get('To', ''),
            'subject': headers.get('Subject', '(no subject)'),
            'date': headers.get('Date', ''),
            'snippet': msg.get('snippet', ''),
            'thread_id': msg.get('threadId', '')
        }

        if full:
            result['body'] = self._extract_body(msg['payload'])

        return result

    def _extract_body(self, payload: dict) -> str:
        """Extract plain text body from email payload."""
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body'].get('data', '')
                    return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                if 'parts' in part:
                    result = self._extract_body(part)
                    if result:
                        return result
        elif payload.get('mimeType') == 'text/plain':
            data = payload['body'].get('data', '')
            return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        return payload.get('snippet', '')

    # ── Draft & Send ──────────────────────────────────────────────────────

    def save_draft(self, to: str, subject: str, body: str) -> str:
        """Save email as draft — returns draft_id."""
        self._ensure_authenticated()
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        draft = self.service.users().drafts().create(
            userId='me',
            body={'message': {'raw': raw}}
        ).execute()
        draft_id = draft['id']
        print(f"[Gmail] Draft saved: {draft_id}")
        return draft_id

    def send_draft(self, draft_id: str) -> bool:
        """Send a saved draft."""
        self._ensure_authenticated()
        self.service.users().drafts().send(
            userId='me',
            body={'id': draft_id}
        ).execute()
        print(f"[Gmail] Draft sent: {draft_id}")
        return True

    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email directly."""
        self._ensure_authenticated()
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        self.service.users().messages().send(
            userId='me',
            body={'raw': raw}
        ).execute()
        print(f"[Gmail] Email sent to {to}")
        return True

    def reply_to_email(self, email_id: str, body: str, save_as_draft: bool = True) -> str:
        """Reply to an existing email — saves as draft by default."""
        self._ensure_authenticated()
        original = self._parse_email(email_id)
        subject = original['subject']
        if not subject.startswith('Re:'):
            subject = f"Re: {subject}"

        message = MIMEText(body)
        message['to'] = original['from']
        message['subject'] = subject
        message['In-Reply-To'] = email_id
        message['References'] = email_id
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        if save_as_draft:
            draft = self.service.users().drafts().create(
                userId='me',
                body={'message': {
                    'raw': raw,
                    'threadId': original['thread_id']
                }}
            ).execute()
            print(f"[Gmail] Reply draft saved: {draft['id']}")
            return draft['id']
        else:
            self.service.users().messages().send(
                userId='me',
                body={'raw': raw, 'threadId': original['thread_id']}
            ).execute()
            return None

    # ── Labels ────────────────────────────────────────────────────────────

    def mark_read(self, email_id: str):
        """Mark email as read."""
        self._ensure_authenticated()
        self.service.users().messages().modify(
            userId='me',
            id=email_id,
            body={'removeLabelIds': ['UNREAD']}
        ).execute()

    def mark_important(self, email_id: str):
        """Mark email as important."""
        self._ensure_authenticated()
        self.service.users().messages().modify(
            userId='me',
            id=email_id,
            body={'addLabelIds': ['IMPORTANT']}
        ).execute()

    # ── Contacts ──────────────────────────────────────────────────────────

    def find_contact_email(self, name: str) -> str | None:
        """Search sent emails to find email address for a name."""
        self._ensure_authenticated()
        results = self.service.users().messages().list(
            userId='me',
            q=f"to:{name}",
            maxResults=5
        ).execute()
        messages = results.get('messages', [])
        if not messages:
            return None
        email = self._parse_email(messages[0]['id'])
        return email.get('to', None)

    # ── Format for speech ─────────────────────────────────────────────────

    def format_emails_for_speech(self, emails: list) -> str:
        """Convert email list to natural speech."""
        if not emails:
            return "No emails found, sir."

        if len(emails) == 1:
            e = emails[0]
            sender = e['from'].split('<')[0].strip()
            return f"One email from {sender} — {e['subject']}. {e['snippet'][:100]}, sir."

        parts = []
        for i, e in enumerate(emails, 1):
            sender = e['from'].split('<')[0].strip()
            parts.append(f"{i}. From {sender} — {e['subject']}")

        return f"You have {len(emails)} emails. " + ". ".join(parts) + ", sir."
