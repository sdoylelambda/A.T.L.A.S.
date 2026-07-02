import asyncio
import glob
import os
from pypdf import PdfReader


# ── PDF finder ────────────────────────────────────────────────────────────────

def find_latest_pdf() -> str | None:
    """Find most recently modified PDF in common locations."""
    search_paths = [
        os.path.expanduser("~/Downloads/*.pdf"),
        os.path.expanduser("~/Documents/*.pdf"),
        os.path.expanduser("~/Desktop/*.pdf"),
        os.path.expanduser("~/Pictures/*.pdf"),
    ]
    all_pdfs = []
    for pattern in search_paths:
        all_pdfs.extend(glob.glob(pattern))

    if not all_pdfs:
        return None

    latest = max(all_pdfs, key=os.path.getmtime)
    print(f"[Documents] Latest PDF: {latest}")
    return latest


def extract_pdf_text(path: str, max_pages: int = 10) -> str | None:
    """Extract plain text from PDF — first N pages."""
    try:
        reader = PdfReader(path)
        total = len(reader.pages)
        pages = min(total, max_pages)
        print(f"[Documents] Reading {pages} of {total} pages")

        text = ""
        for i in range(pages):
            page_text = reader.pages[i].extract_text() or ""
            text += page_text

        if not text.strip():
            print("[Documents] No text extracted — may be scanned/image PDF")
            return None

        print(f"[Documents] Extracted {len(text)} characters")
        return text

    except Exception as e:
        print(f"[Documents] PDF read error: {e}")
        return None


def summarize_text(text: str, brain, instruction: str = None, use_gemini: bool = False, bypass_permission: bool = False) -> str:
    """Summarize document text — Mistral locally or Gemini for deep analysis."""
    max_chars = 50000 if use_gemini else 8000
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
        print(f"[Documents] Truncated to {max_chars} chars")

    focus = f"\nFocus on: {instruction}" if instruction else ""
    trunc_note = "\n(Note: document was truncated due to length)" if truncated else ""

    prompt = f"""Summarize this document concisely for a voice assistant.
3-5 sentences max. Plain text only. No markdown. End with 'sir'.{focus}{trunc_note}

Document:
{text}

Summary:"""

    model = "gemini" if use_gemini else "orchestrator"
    ctx = None if use_gemini else 8192
    print(f"[Documents] Summarizing with {model}")

    try:
        return brain.query(
            prompt,
            model_key=model,
            num_ctx_override=ctx,
            max_tokens_override=300,
            bypass_permission=bypass_permission
        )
    except Exception as e:
        print(f"[Documents] {model} error: {e}")
        if use_gemini:
            print("[Documents] Gemini failed, falling back to Mistral")
            return summarize_text(text, brain, instruction, use_gemini=False)
        return "I had trouble summarizing that document, sir."


# ── Main handler ──────────────────────────────────────────────────────────────

async def handle_document_command(text: str, face, mouth, brain, ears=None, stt=None, say=None, debug: bool = False) -> bool:
    """
    Handle document commands. Returns True if handled.
    """

    # detect deep analysis intent
    use_gemini = any(phrase in text for phrase in [
        "gemini", "detailed", "deep", "thorough",
        "take your time", "in detail", "full analysis",
        "detailed summary", "comprehensive"
    ])

    if use_gemini:
        if say:
            await say(
                "This will send the document to Gemini for analysis. "
                "Shall I proceed, sir?"
            )
        else:
            await _say(face, mouth,
                       "This will send the document to Gemini. Shall I proceed, sir?")

        confirmed = await _listen_for_yes(ears, stt)
        if not confirmed:
            await _say(face, mouth, "Using local analysis instead, sir.")
            use_gemini = False

    # ── PDF summarization ─────────────────────────────────────────────────
    if any(phrase in text for phrase in [
        "summarize pdf", "summarize the pdf", "read the pdf", "read me the pdf",
        "what does the pdf say", "summarize last pdf",
        "what's in the pdf", "read this pdf", "summarize document",
        "what's in the document", "read the document"
    ]):
        face.set_caption("finding PDF...")

        path = await asyncio.to_thread(find_latest_pdf)
        if not path:
            await _say(face, mouth,
                "No PDF files found, sir. Try putting one in Downloads or Documents.")
            return True

        face.set_caption("reading PDF...")
        pdf_text = await asyncio.to_thread(extract_pdf_text, path)
        if not pdf_text:
            await _say(face, mouth,
                "Couldn't extract text from that PDF, sir. "
                "It may be a scanned image — try a text-based PDF.")
            return True

        if any(phrase in text for phrase in [
            "read me the pdf",
            ]):
            chunks = [
                pdf_text[i:i + 2000]
                for i in range(0, len(pdf_text), 2000)
            ]

            for chunk in chunks:
                await _say(face, mouth, chunk)

            return True

        # extract focus instruction if any
        instruction = None
        for phrase in ["focus on", "looking for", "about", "find"]:
            if phrase in text:
                instruction = text.split(phrase)[-1].strip()
                break

        caption = "deep analysis with Gemini..." if use_gemini else "summarizing..."
        face.set_caption(caption)

        summary = await asyncio.to_thread(
            summarize_text, pdf_text, brain, instruction, use_gemini, True
        )
        await _say(face, mouth, summary, next_state="listening")
        return True

    # ── Specific PDF question ─────────────────────────────────────────────
    if any(phrase in text for phrase in [
        "what does the document say about",
        "find in the pdf", "search the pdf",
        "look for in the document"
    ]):
        face.set_caption("searching PDF...")

        path = await asyncio.to_thread(find_latest_pdf)
        if not path:
            await _say(face, mouth, "No PDF files found, sir.")
            return True

        pdf_text = await asyncio.to_thread(extract_pdf_text, path)
        if not pdf_text:
            await _say(face, mouth, "Couldn't read that PDF, sir.")
            return True

        face.set_caption("searching...")
        summary = await asyncio.to_thread(
            summarize_text, pdf_text, brain, text, use_gemini
        )
        await _say(face, mouth, summary, next_state="listening")
        return True

    return False


async def _say(face, mouth, text: str, next_state: str = "listening") -> None:
    """Minimal say for document handler."""
    from modules.utils import timer
    face.set_caption(text)
    face.set_state("speaking")
    with timer("TTS doc", True):
        await mouth.speak(text)
    face.set_state(next_state)

async def _listen_for_yes(ears, stt) -> bool:
    """Listen for a yes/no confirmation."""
    if not ears or not stt:
        print("[Documents] No ears/stt available")
        return False
    try:
        print("[Documents] Listening for confirmation...")
        await asyncio.sleep(1.5)
        audio_bytes, dur = await ears.listen()
        response = stt.transcribe(audio_bytes, dur).lower().strip() if audio_bytes else ""
        print(f"[Documents] Heard: '{response}'")
        return any(w in response for w in [
            "yes", "yeah", "yep", "sure", "do it",
            "proceed", "go ahead", "affirmative", "send it"
        ])
    except Exception as e:
        print(f"[Documents] Listen error: {e}")
        return False
