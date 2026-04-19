"""Azure OpenAI client bootstrap.

Reads four env vars:

    AZURE_OPENAI_API_KEY       — required
    AZURE_OPENAI_ENDPOINT      — e.g. https://instance-kr01.openai.azure.com
    AZURE_OPENAI_DEPLOYMENT    — deployment name (e.g. gpt-5-mini); passed as
                                 ``model`` in ``chat.completions.create``
    AZURE_OPENAI_API_VERSION   — defaults to 2024-10-21
"""

from __future__ import annotations

import os

from openai import AzureOpenAI

_DEFAULT_API_VERSION = "2024-10-21"


class AzureConfigError(RuntimeError):
    """Raised when required Azure env vars are missing."""


def get_client() -> AzureOpenAI:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        raise AzureConfigError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must both be set"
        )
    return AzureOpenAI(
        api_key=api_key,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", _DEFAULT_API_VERSION),
        azure_endpoint=endpoint,
    )


def get_deployment() -> str:
    dep = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not dep:
        raise AzureConfigError("AZURE_OPENAI_DEPLOYMENT must be set")
    return dep
