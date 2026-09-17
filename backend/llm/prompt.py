"""Prompt assembly for the inventory voice agent.

Two properties matter here:

1. The prompt names the reply language explicitly. Listings are stored in
   English, so without that instruction the model answers in English even
   when asked in Hindi — and the Hindi TTS voice then renders English text as
   unintelligible audio. This is where the multilingual feature is won or lost.

2. The token budget is constant. Retrieval caps the listings and the history
   window caps the conversation, so a 50-turn call costs the same per turn as
   a 2-turn call regardless of how large the catalog grows.
"""

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

_BASE_PROMPT = (
    "You are a phone assistant for a store. Answer only using the listings "
    "given below. If no listings are given, say there are no matching "
    "items in stock. Be brief, spoken-language style, one or two sentences. "
)

# For a non-English caller the contrast has to be spelled out, or the model
# follows the language of its context (English listings) instead of the
# caller's. For an English caller that same sentence is redundant and reads
# as contradictory, so it is dropped.
_REPLY_IN_ENGLISH = "Reply in English."
_REPLY_IN_OTHER = "Reply in {language}, even though the listings are written in English."


def _system_prompt(language_name: str) -> str:
    if language_name == "English":
        return _BASE_PROMPT + _REPLY_IN_ENGLISH
    return _BASE_PROMPT + _REPLY_IN_OTHER.format(language=language_name)


def _format_items(items: list[dict]) -> str:
    if not items:
        return "No matching listings found."
    # Essential fields only — the full description is deliberately dropped,
    # roughly halving the tokens each listing costs.
    return "\n".join(
        f"- {item['name']}: ${item['price']}, stock {item['stock']}, "
        f"category {item['category']}"
        for item in items
    )


def _format_history(history: list[dict], max_turns: int) -> str:
    windowed = history[-max_turns:] if max_turns else history
    return "\n".join(f"{turn['role']}: {turn['text']}" for turn in windowed)


def build_prompt(user_text: str, retrieved: list[dict], history: list[dict],
                 language: str = "en", max_history_turns: int = 6) -> str:
    language_name = LANGUAGE_NAMES.get(language, "English")
    parts = [
        _system_prompt(language_name),
        "",
        "Relevant listings:",
        _format_items(retrieved),
    ]
    history_text = _format_history(history, max_history_turns)
    if history_text:
        parts += ["", "Conversation so far:", history_text]
    parts += ["", f"user: {user_text}", "assistant:"]
    return "\n".join(parts)
