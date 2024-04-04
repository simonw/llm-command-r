import llm
from pydantic import Field, field_validator, model_validator
from typing import Optional, List
import cohere


@llm.hookimpl
def register_models(register):
    # https://docs.cohere.com/docs/models
    register(CohereMessages("command-r"), aliases=("r",))
    register(CohereMessages("command-r-plus"), aliases=("r-plus",))


class CohereMessages(llm.Model):
    needs_key = "cohere"
    key_env_var = "COHERE_API_KEY"
    can_stream = True

    class Options(llm.Options):
        websearch: Optional[bool] = Field(
            description="Use web search connector",
            default=False,
        )

    def __init__(self, model_id):
        self.model_id = model_id

    def build_chat_history(self, conversation) -> List[dict]:
        chat_history = []
        if conversation:
            for response in conversation.responses:
                chat_history.extend(
                    [
                        {"role": "USER", "text": response.prompt.prompt},
                        {"role": "CHATBOT", "text": response.text()},
                    ]
                )
        return chat_history

    def execute(self, prompt, stream, response, conversation):
        client = cohere.Client(self.get_key())
        kwargs = {
            "message": prompt.prompt,
            "model": self.model_id,
        }
        if prompt.system:
            kwargs["preamble"] = prompt.system

        if conversation:
            kwargs["chat_history"] = self.build_chat_history(conversation)

        if prompt.options.websearch:
            kwargs["connectors"] = [{"id": "web-search"}]

        if stream:
            for event in client.chat_stream(**kwargs):
                if event.event_type == "text-generation":
                    yield event.text
                elif event.event_type == "stream-end":
                    response.response_json = dict(event.response)
        else:
            event = client.chat(**kwargs)
            answer = event.text
            yield answer
            response.response_json = dict(event)

    def __str__(self):
        return "Cohere Messages: {}".format(self.model_id)
