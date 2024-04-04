# llm-command-r

[![PyPI](https://img.shields.io/pypi/v/llm-command-r.svg)](https://pypi.org/project/llm-command-r/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-command-r?include_prereleases&label=changelog)](https://github.com/simonw/llm-command-r/releases)
[![Tests](https://github.com/simonw/llm-command-r/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/llm-command-r/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-command-r/blob/main/LICENSE)

Access the [Cohere Command R](https://docs.cohere.com/docs/command-r) family of models via the Cohere API

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-command-r
```

## Configuration

You will need a [Cohere API key](https://dashboard.cohere.com/api-keys). Configure it like this:

```bash
llm keys set cohere
# Paste key here
```

## Usage

This plugin adds two models.

```bash
llm -m command-r 'Say hello from Command R'
llm -m command-r-plus 'Say hello from Command R Plus'
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-command-r
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
pytest
```
