import llm


def test_plugin_is_installed():
    model_ids = [m.model_id for m in llm.get_models()]
    assert "command-r" in model_ids
