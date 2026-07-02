import asyncio
from modules.utils import timer


async def handle_vision_command(text: str, face, mouth, eyes, debug: bool = False) -> bool:
    """
    Handle vision commands. Returns True if handled, False if not a vision command.
    """
    if not eyes:
        return False

    if debug:
        print(f"[Eyes] checking: '{text}'")

    async def analyze(caption: str, analyze_fn) -> None:
        face.set_caption(caption)
        warning = eyes.check_storage()
        if warning:
            await _say(face, mouth, warning)
        face.set_state("thinking")
        result = await asyncio.to_thread(analyze_fn)
        await _say(face, mouth, result, next_state="listening")

    if any(phrase in text for phrase in [
        "what do you see", "what can you see", "what's in front",
        "describe what you see", "look around", "describe the scene",
        "what's in the room", "what's around", "describe my workspace",
        "what's on my desk", "what's behind me", "take a picture", "take a photo"
    ]):
        await analyze("looking...", eyes.describe_scene)
        return True

    if any(phrase in text for phrase in [
        "what am i holding", "what is this", "identify this",
        "what's this object", "what object is this"
    ]):
        await analyze("identifying...", eyes.identify_object)
        return True

    if any(phrase in text for phrase in [
        "read this", "what does this say", "read the text",
        "what's written", "transcribe this", "read this document",
        "what does this paper say", "read the text on screen"
    ]):
        await analyze("reading...", eyes.read_document)
        return True

    if any(phrase in text for phrase in [
        "is anyone there", "is someone there", "anyone in the room",
        "is there someone", "who's there", "who is in the room",
        "how many people", "is anyone looking", "anyone here"
    ]):
        await analyze("checking...", eyes.count_people)
        return True

    if any(phrase in text for phrase in [
        "what am i doing", "what are they doing", "what's happening",
        "describe the activity", "what is this person doing"
    ]):
        await analyze("observing...", eyes.describe_activity)
        return True

    if any(phrase in text for phrase in [
        "what color is this", "what colour is this",
        "what color am i", "what colour am i"
    ]):
        await analyze("analyzing color...", eyes.identify_color)
        return True

    if any(phrase in text for phrase in [
        "do you see", "can you see", "is there a", "is there an",
        "does this look", "does this seem", "what does this look like"
    ]):
        await analyze("looking...", lambda: eyes.analyze_with_question(text))
        return True

    # screen capture
    if any(phrase in text for phrase in [
        "take a look at my screen", "what's on my screen", "whats on my screen"
        "look at my screen", "analyze my screen",
        "what am i looking at", "what's on screen"
    ]):
        face.set_caption("capturing screen...")
        result = await eyes.analyze_screen()
        await _say(face, mouth, result, next_state="listening")
        return True

    # latest screenshot
    if any(phrase in text for phrase in [
        "analyze last screenshot", "look at my screenshot",
        "analyze screenshot", "what's in my screenshot",
        "last screenshot", "recent screenshot"
    ]):
        face.set_caption("analyzing screenshot...")
        result = await eyes.analyze_screenshot()
        await _say(face, mouth, result, next_state="listening")
        return True

    # screen + specific question
    if "screen" in text and any(phrase in text for phrase in [
        "what is", "explain", "describe", "summarize", "what does"
    ]):
        face.set_caption("analyzing screen...")
        result = await eyes.analyze_screen()
        await _say(face, mouth, result, next_state="listening")
        return True

    return False


async def _say(face, mouth, text: str, next_state: str = None) -> None:
    """Minimal say for vision module — no ear management."""
    face.set_caption(text)
    face.set_state("speaking")
    with timer("TTS vision", True):
        await mouth.speak(text)
    face.set_state(next_state or "listening")
