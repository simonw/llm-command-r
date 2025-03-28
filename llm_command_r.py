import click
import cohere
import llm
import os
from pydantic import Field, ValidationError
import sqlite_utils
import sys
from typing import Optional, List


@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("prompt")
    @click.option("-s", "--system", help="System prompt to use")
    @click.option("model_id", "-m", "--model", help="Model to use")
    @click.option(
        "options",
        "-o",
        "--option",
        type=(str, str),
        multiple=True,
        help="key/value options for the model",
    )
    @click.option("-n", "--no-log", is_flag=True, help="Don't log to database")
    @click.option("--key", help="API key to use")
    def command_r_search(prompt, system, model_id, options, no_log, key):
        "Prompt Command R with the web search feature"
        from llm.cli import logs_on, logs_db_path
        from llm.migrations import migrate

        model_id = model_id or "command-r"
        model = llm.get_model(model_id)
        if model.needs_key:
            model.key = llm.get_key(key, model.needs_key, model.key_env_var)
        validated_options = {}
        options = list(options)
        options.append(("websearch", "1"))
        try:
            validated_options = dict(
                (key, value)
                for key, value in model.Options(**dict(options))
                if value is not None
            )
        except ValidationError as ex:
            raise click.ClickException(ex.errors())
        response = model.prompt(prompt, system=system, **validated_options)
        for chunk in response:
            print(chunk, end="")
            sys.stdout.flush()

        # Log to the database
        if logs_on() and not no_log:
            log_path = logs_db_path()
            (log_path.parent).mkdir(parents=True, exist_ok=True)
            db = sqlite_utils.Database(log_path)
            migrate(db)
            response.log_to_db(db)

        # Now output the citations
        documents = response.response_json.get("documents", [])
        if documents:
            print()
            print()
            print("Sources:")
            print()
            for doc in documents:
                print("-", doc["title"], "-", doc["url"])


@llm.hookimpl
def register_models(register):
    # https://docs.cohere.com/docs/models
    register(CohereMessages("command-r"), aliases=("r",))
    register(CohereMessages("command-r-plus"), aliases=("r-plus",))
    register(CohereMessages("command-r7b-12-2024"), aliases=("command-r7b",))
    register(CohereMessages("command-a-03-2025"), aliases=("command-a",))


class CohereMessages(llm.Model):
    needs_key = "cohere"
    key_env_var = "COHERE_API_KEY"
    can_stream = True
    supports_schema = True

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
        kwargs = {"log_warning_experimental_features": False}
        if os.environ.get("COHERE_BASE_URL"):
            kwargs["base_url"] = os.environ["COHERE_BASE_URL"]
        client = cohere.Client(self.get_key(), **kwargs)
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

        if prompt.schema:
            kwargs["response_format"] = {"type": "json_object", "schema": prompt.schema}

        if stream:
            for event in client.chat_stream(**kwargs):
                if event.event_type == "text-generation":
                    yield event.text
                elif event.event_type == "stream-end":
                    response.response_json = event.response.dict()
        else:
            event = client.chat(**kwargs)
            answer = event.text
            yield answer
            response.response_json = event.dict()

    def __str__(self):
        return "Cohere Messages: {}".format(self.model_id)
