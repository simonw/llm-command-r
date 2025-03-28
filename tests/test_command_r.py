import llm
import os
import pytest

COHERE_API_KEY = os.environ.get("PYTEST_COHERE_API_KEY", None) or "sk-..."


@pytest.mark.vcr
def test_prompt():
    model = llm.get_model("command-r")
    model.key = model.key or COHERE_API_KEY
    response = model.prompt("Two names for a pet pelican, be brief")
    assert str(response) == "Pelly and Mr. P"
    response_dict = dict(response.response_json)
    response_dict.pop("generation_id")  # differs between requests
    response_dict.pop("response_id")
    assert response_dict == {
        "text": "Pelly and Mr. P",
        "finish_reason": "COMPLETE",
        "chat_history": [
            {"role": "USER", "message": "Two names for a pet pelican, be brief"},
            {"role": "CHATBOT", "message": "Pelly and Mr. P"},
        ],
        "meta": {
            "api_version": {"version": "1"},
            "billed_units": {"input_tokens": 10.0, "output_tokens": 6.0},
            "tokens": {"input_tokens": 76.0, "output_tokens": 6.0},
        },
    }
