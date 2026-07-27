"""The Anthropic <-> OpenAI translation, which is the part of this proxy that
breaks silently: a mistranslated tool call doesn't raise, it just makes Claude
Code look like it forgot how to edit files."""

import json

import main


# ---------------------------------------------------------------------------
# Requests: Anthropic -> OpenAI
# ---------------------------------------------------------------------------

def test_system_string_and_blocks():
    assert main.system_to_text("be brief") == "be brief"
    assert main.system_to_text([{"type": "text", "text": "a"},
                                {"type": "text", "text": "b"}]) == "a\n\nb"
    assert main.system_to_text(None) is None


def test_plain_messages_carry_through():
    out = main.anthropic_messages_to_openai("sys", [{"role": "user", "content": "hi"}])
    assert out == [{"role": "system", "content": "sys"},
                   {"role": "user", "content": "hi"}]


def test_tool_use_becomes_openai_tool_calls():
    out = main.anthropic_messages_to_openai(None, [{
        "role": "assistant",
        "content": [{"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                     "input": {"path": "a.py"}}],
    }])
    assert out[0]["content"] == "checking"
    call = out[0]["tool_calls"][0]
    assert call["id"] == "toolu_1"
    assert call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "a.py"}


def test_tool_result_becomes_its_own_tool_message():
    out = main.anthropic_messages_to_openai(None, [{
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "file body"}],
    }])
    assert out == [{"role": "tool", "tool_call_id": "toolu_1", "content": "file body"}]


def test_tool_result_block_list_is_flattened():
    out = main.anthropic_messages_to_openai(None, [{
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t", "content": [
            {"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}],
    }])
    assert out[0]["content"] == "one\ntwo"


def test_image_blocks_become_a_placeholder():
    out = main.anthropic_messages_to_openai(None, [{
        "role": "user", "content": [{"type": "image", "source": {}}],
    }])
    assert "image omitted" in out[0]["content"]


def test_tools_and_tool_choice():
    tools = main.anthropic_tools_to_openai([
        {"name": "bash", "description": "run", "input_schema": {"type": "object", "properties": {}}}])
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "bash"

    assert main.anthropic_tool_choice_to_openai({"type": "auto"}) == "auto"
    assert main.anthropic_tool_choice_to_openai({"type": "any"}) == "required"
    assert main.anthropic_tool_choice_to_openai(
        {"type": "tool", "name": "bash"})["function"]["name"] == "bash"
    assert main.anthropic_tool_choice_to_openai(None) is None


# ---------------------------------------------------------------------------
# Responses: OpenAI -> Anthropic
# ---------------------------------------------------------------------------

def test_text_response():
    out = main.openai_response_to_anthropic(
        {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
        "claude-sonnet-4")
    assert out["content"] == [{"type": "text", "text": "hello"}]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"] == {"input_tokens": 7, "output_tokens": 3}


def test_tool_calls_become_tool_use():
    out = main.openai_response_to_anthropic({"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "call_1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
        "finish_reason": "tool_calls"}]}, "m")
    block = out["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "bash"
    assert block["input"] == {"command": "ls"}
    assert out["stop_reason"] == "tool_use"


def test_empty_choices_is_not_a_crash():
    """A content filter (and some error shapes) return choices: []. Indexing
    [0] there used to be an unhandled IndexError -- a 500 on /v1/messages."""
    out = main.openai_response_to_anthropic({"choices": []}, "m")
    assert out["content"] == [{"type": "text", "text": ""}]
    assert out["stop_reason"] == "end_turn"

    assert main.openai_response_to_anthropic({}, "m")["content"] == [{"type": "text", "text": ""}]


def test_object_arguments_do_not_raise():
    """llama.cpp's server and some vLLM builds send `arguments` as an object
    rather than a JSON string. json.loads on a dict raises TypeError, which
    the old handler did not catch."""
    out = main.openai_response_to_anthropic({"choices": [{"message": {"tool_calls": [
        {"id": "c", "function": {"name": "bash", "arguments": {"command": "ls"}}}]}}]}, "m")
    assert out["content"][0]["input"] == {"command": "ls"}


def test_unparseable_arguments_become_empty_input():
    out = main.openai_response_to_anthropic({"choices": [{"message": {"tool_calls": [
        {"id": "c", "function": {"name": "bash", "arguments": "{not json"}}]}}]}, "m")
    assert out["content"][0]["input"] == {}


def test_nameless_tool_call_is_dropped_not_emitted():
    """A tool_use block with no name is rejected by Claude Code, so passing it
    through turns one bad provider chunk into a failed turn."""
    out = main.openai_response_to_anthropic({"choices": [{"message": {"content": "hi", "tool_calls": [
        {"id": "c", "function": {"arguments": "{}"}}]}}]}, "m")
    assert [b["type"] for b in out["content"]] == ["text"]


def test_missing_tool_call_id_gets_generated():
    out = main.openai_response_to_anthropic({"choices": [{"message": {"tool_calls": [
        {"function": {"name": "bash", "arguments": "{}"}}]}}]}, "m")
    assert out["content"][0]["id"].startswith("toolu_")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_tool_args_dict_accepts_every_shape():
    assert main.tool_args_dict('{"a": 1}') == {"a": 1}
    assert main.tool_args_dict({"a": 1}) == {"a": 1}
    assert main.tool_args_dict(None) == {}
    assert main.tool_args_dict("") == {}
    assert main.tool_args_dict("[1, 2]") == {}       # valid JSON, wrong type
    assert main.tool_args_dict("{oops") == {}


def test_tool_args_str_serialises_objects():
    assert main.tool_args_str('{"a": 1}') == '{"a": 1}'
    assert json.loads(main.tool_args_str({"a": 1})) == {"a": 1}
    assert main.tool_args_str(None) == ""


def test_first_choice_guards():
    assert main.first_choice({"choices": [{"x": 1}]}) == {"x": 1}
    assert main.first_choice({"choices": []}) == {}
    assert main.first_choice({"choices": "nonsense"}) == {}
    assert main.first_choice({}) == {}
    assert main.first_choice(None) == {}


def test_output_tokens_prefers_the_provider():
    assert main._output_tokens({"completion_tokens": 42}, 4000) == 42
    assert main._output_tokens(None, 400) == 100          # chars // 4
    assert main._output_tokens({"completion_tokens": "x"}, 40) == 10


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

def test_pick_model_buckets_on_haiku():
    backend = {"big_model": "big", "small_model": "small"}
    assert main.pick_model("claude-3-5-haiku-latest", backend) == "small"
    assert main.pick_model("claude-sonnet-4", backend) == "big"
    assert main.pick_model("claude-opus-4", backend) == "big"
    assert main.pick_model("", backend) == "big"
    assert main.pick_model(None, backend) == "big"


def test_pick_model_survives_a_hand_edited_backend():
    """backends.json is hand-editable and the README encourages editing it.
    A missing small_model used to be a KeyError on every haiku request."""
    assert main.pick_model("claude-3-5-haiku", {"big_model": "only"}) == "only"
    assert main.pick_model("claude-sonnet-4", {"small_model": "only"}) == "only"
    assert main.pick_model("claude-sonnet-4", {}) == ""


def test_build_payload_maps_the_optional_fields():
    payload = main.build_openai_payload({
        "model": "claude-3-5-haiku", "max_tokens": 64, "temperature": 0.2,
        "top_p": 0.9, "stop_sequences": ["END"],
        "messages": [{"role": "user", "content": "hi"}],
    }, {"big_model": "big", "small_model": "small"})
    assert payload["model"] == "small"
    assert payload["max_tokens"] == 64
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9
    assert payload["stop"] == ["END"]
