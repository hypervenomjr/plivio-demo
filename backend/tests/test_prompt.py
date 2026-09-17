from backend.llm.prompt import build_prompt

RETRIEVED = [
    {"id": "1", "name": "Juniper Active Noise Cancelling Wireless Headphones",
     "description": "...", "price": 89.99, "stock": 20,
     "category": "Electronics", "score": 0.1},
]


def test_prompt_includes_user_text():
    prompt = build_prompt("do you have headphones", RETRIEVED, [])
    assert "do you have headphones" in prompt


def test_prompt_includes_retrieved_item_essential_fields():
    prompt = build_prompt("headphones?", RETRIEVED, [])
    assert "Juniper Active Noise Cancelling Wireless Headphones" in prompt
    assert "89.99" in prompt
    assert "20" in prompt


def test_prompt_omits_retrieved_description_to_save_tokens():
    retrieved = [dict(RETRIEVED[0], description="A" * 500)]
    prompt = build_prompt("headphones?", retrieved, [])
    assert "A" * 500 not in prompt


def test_prompt_states_no_listings_when_retrieval_empty():
    prompt = build_prompt("headphones?", [], [])
    assert "No matching listings found." in prompt


def test_prompt_windows_history_to_max_turns():
    history = [{"role": "user", "text": f"turn {i}"} for i in range(10)]
    prompt = build_prompt("latest question", [], history, max_history_turns=2)
    assert "turn 9" in prompt
    assert "turn 0" not in prompt


def test_prompt_omits_history_section_when_empty():
    prompt = build_prompt("hello", [], [])
    assert "Conversation so far:" not in prompt


def test_prompt_instructs_reply_language_hindi():
    prompt = build_prompt("क्या हेडफ़ोन हैं", RETRIEVED, [], language="hi")
    assert "Hindi" in prompt


def test_prompt_instructs_reply_language_marathi():
    prompt = build_prompt("हेडफोन आहेत का", RETRIEVED, [], language="mr")
    assert "Marathi" in prompt


def test_prompt_defaults_to_english():
    prompt = build_prompt("do you have headphones", RETRIEVED, [])
    assert "English" in prompt


def test_prompt_falls_back_to_english_for_unknown_language():
    prompt = build_prompt("hello", RETRIEVED, [], language="xx")
    assert "English" in prompt


def test_prompt_stays_within_token_budget():
    """The prompt must not grow with catalog size or call length.

    Five listings plus six history turns should stay well under the 4096-token
    context, leaving room for generation. Rough proxy: 4 chars per token.
    """
    retrieved = [
        dict(RETRIEVED[0], id=str(i), name=f"Brand{i} Wireless Headphones")
        for i in range(5)
    ]
    history = [{"role": "user", "text": "a fairly typical spoken question here"}
               for _ in range(6)]
    prompt = build_prompt("what about in blue", retrieved, history)
    assert len(prompt) / 4 < 800
