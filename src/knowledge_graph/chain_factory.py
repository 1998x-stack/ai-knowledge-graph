"""Factory utilities for constructing LangChain chat pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


@dataclass(frozen=True)
class LLMSettings:
    """Configuration for creating chat model instances."""

    model: str
    api_key: str
    max_tokens: int
    temperature: float
    base_url: str | None = None


@dataclass(frozen=True)
class PromptTemplateConfig:
    """Container describing a chat prompt template."""

    name: str
    system_template: str
    user_template: str
    chain_type: str = "chat-generic"


class PluggableChainFactory:
    """Factory that creates chat pipelines using registered builders."""

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[[LLMSettings, PromptTemplateConfig], Runnable]] = {}

    def register(self, name: str, builder: Callable[[LLMSettings, PromptTemplateConfig], Runnable]) -> None:
        """Register a builder function for a given chain type."""

        self._registry[name] = builder

    def create(self, name: str, settings: LLMSettings, prompt: PromptTemplateConfig) -> Runnable:
        """Create a runnable chain for the given settings and prompt."""

        try:
            builder = self._registry[name]
        except KeyError as exc:
            raise ValueError(f"Unknown chain type '{name}'") from exc
        return builder(settings, prompt)


def _build_chat_chain(settings: LLMSettings, prompt: PromptTemplateConfig) -> Runnable:
    """Build a simple chat pipeline that returns plain text responses."""

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt.system_template),
            ("user", prompt.user_template),
        ]
    )

    chat_model = ChatOpenAI(
        model=settings.model,
        openai_api_key=settings.api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        openai_api_base=settings.base_url,
    )

    return chat_prompt | chat_model | StrOutputParser()


def default_factory() -> PluggableChainFactory:
    """Create a factory instance with default builders registered."""

    factory = PluggableChainFactory()
    factory.register("chat-generic", _build_chat_chain)
    return factory

