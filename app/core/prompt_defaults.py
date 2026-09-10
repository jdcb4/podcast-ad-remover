"""Shared legacy defaults for execution, display and explicit reset."""

LEGACY_DEFAULTS = {
    "ad_base": '''Identify segments in the transcript that match the Targets.
Targets: {targets}
{custom_instr}
Return a JSON array of objects with "start", "end", "label" (Ad/Promo/Intro/Outro), and "reason" (brief explanation).
Example: [{"start": 0.0, "end": 10.0, "label": "Ad", "reason": "Sponsor read for XYZ"}]''',
    "sponsor": "Sponsor messages",
    "promo": "Promos",
    "intro": "Intro",
    "outro": "Outro",
    "summary": '''You are a smart assistant. Write a short 2-3 sentence summary of this podcast episode.
The summary must:
1. NOT mention the podcast name, episode title, or date.
2. Start immediately with "This episode includes".
3. Briefly summarize key topics.
Transcript Context: {transcript_context}''',
}
