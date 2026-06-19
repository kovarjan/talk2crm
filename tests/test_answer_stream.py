import pytest
from app.engine.answer_stream import AnswerStreamState


def collect(state: AnswerStreamState, chunks: list[str]) -> str:
    """Feed all chunks and flush; return all emitted text concatenated."""
    out = ""
    for chunk in chunks:
        out += state.feed(chunk)
    out += state.flush()
    return out


# --- tool call suppression ---

def test_tool_call_is_suppressed():
    state = AnswerStreamState()
    result = collect(state, ['<tool_call>{"name":"crm_query_tool"}</tool_call>'])
    assert result == ""
    assert state.is_suppressed


def test_tool_call_across_chunks_is_suppressed():
    state = AnswerStreamState()
    result = collect(state, ["<tool", "_call>", '{"name":"x"}', "</tool_call>"])
    assert result == ""
    assert state.is_suppressed


# --- answer tag streaming ---

def test_answer_tag_streams_content():
    state = AnswerStreamState()
    result = collect(state, ["<answer>Hello world</answer>"])
    assert result == "Hello world"
    assert not state.is_suppressed


def test_answer_tag_split_across_chunks():
    state = AnswerStreamState()
    result = collect(state, ["<answ", "er>", "Hi", " there", "</answer>"])
    assert result == "Hi there"


def test_plain_text_streams_as_answer():
    state = AnswerStreamState()
    result = collect(state, ["Nalezl jsem kontakt."])
    assert result == "Nalezl jsem kontakt."
    assert not state.is_suppressed


# --- think block suppression ---

def test_think_block_stripped_before_answer():
    state = AnswerStreamState()
    result = collect(state, ["<think>reasoning</think><answer>Final answer</answer>"])
    assert result == "Final answer"


def test_think_block_split_across_chunks():
    state = AnswerStreamState()
    result = collect(state, ["<think>", "some thought", "</think>", "<answer>", "Done", "</answer>"])
    assert result == "Done"


def test_think_then_tool_call_suppressed():
    state = AnswerStreamState()
    result = collect(state, ["<think>reasoning</think>", '<tool_call>{"name":"x"}</tool_call>'])
    assert result == ""
    assert state.is_suppressed


# --- tail guard ---

def test_tail_guard_holds_closing_tag():
    state = AnswerStreamState()
    chunks = ["<answer>", "A" * 50, "</answer>"]
    result = collect(state, chunks)
    assert "</answer>" not in result
    assert "A" * 50 in result


def test_flush_returns_remaining_content():
    state = AnswerStreamState()
    emitted_during_feed = state.feed("Hi")
    assert emitted_during_feed == ""  # held in tail guard
    tail = state.flush()
    assert tail == "Hi"


# --- independent state instances ---

def test_two_instances_are_independent():
    s1 = AnswerStreamState()
    s2 = AnswerStreamState()
    collect(s1, ["<tool_call>x</tool_call>"])
    result = collect(s2, ["<answer>ok</answer>"])
    assert s1.is_suppressed
    assert not s2.is_suppressed
    assert result == "ok"
