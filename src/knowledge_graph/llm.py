"""LLM utilities built on top of LangChain."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.runnables import Runnable

from src.knowledge_graph.chain_factory import LLMSettings, PluggableChainFactory, default_factory
from src.knowledge_graph.prompts import PromptRegistry


@dataclass
class LLMService:
    """Service wrapper responsible for orchestrating LangChain pipelines."""

    settings: LLMSettings
    factory: PluggableChainFactory
    prompts: PromptRegistry
    _chain_cache: Dict[str, Runnable]

    def __init__(
        self,
        settings: LLMSettings,
        factory: PluggableChainFactory | None = None,
        prompts: PromptRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.factory = factory or default_factory()
        self.prompts = prompts or PromptRegistry()
        self._chain_cache = {}

    def invoke(self, prompt_name: str, variables: Dict[str, Any], debug: bool = False) -> str:
        """Execute the registered chain and return the raw text response."""

        chain = self._get_chain(prompt_name)
        response = chain.invoke(variables)

        if debug:
            print("Raw LLM response:")
            print(response)
            print("\n---\n")

        return response

    def _get_chain(self, prompt_name: str) -> Runnable:
        """Return a cached chain instance for the requested prompt."""

        if prompt_name not in self._chain_cache:
            prompt_config = self.prompts.get(prompt_name)
            self._chain_cache[prompt_name] = self.factory.create(
                prompt_config.chain_type,
                self.settings,
                prompt_config,
            )
        return self._chain_cache[prompt_name]


def service_from_config(config: Dict[str, Any]) -> LLMService:
    """Instantiate :class:`LLMService` using configuration dictionary values."""

    llm_config = config.get("llm", {})

    required_fields = {"model", "api_key"}
    missing = [field for field in required_fields if field not in llm_config]
    if missing:
        raise ValueError(f"Missing LLM configuration fields: {', '.join(sorted(missing))}")

    settings = LLMSettings(
        model=llm_config["model"],
        api_key=llm_config["api_key"],
        max_tokens=int(llm_config.get("max_tokens", 1000)),
        temperature=float(llm_config.get("temperature", 0.2)),
        base_url=llm_config.get("base_url"),
    )

    return LLMService(settings)


def extract_json_from_text(text: str) -> Any:
    """Extract JSON data from free-form text produced by an LLM."""

    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    code_match = re.search(code_block_pattern, text)
    if code_match:
        text = code_match.group(1).strip()
        print("Found JSON in code block, extracting content...")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_idx = text.find("[")
        if start_idx == -1:
            print("No JSON array start found in text")
            return None

        bracket_count = 0
        complete_json = False
        json_str = ""
        for i, char in enumerate(text[start_idx:], start=start_idx):
            if char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    json_str = text[start_idx : i + 1]
                    complete_json = True
                    break

        if complete_json:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                print("Found JSON-like structure but couldn't parse it. Attempting repairs...")
                fixed_json = re.sub(r"(\s*)(\w+)(\s*):(\s*)", r'\1"\2"\3:\4', json_str)
                fixed_json = re.sub(r",(\s*[\]}])", r"\1", fixed_json)
                try:
                    return json.loads(fixed_json)
                except json.JSONDecodeError:
                    print("Could not fix JSON format issues")
        else:
            print("Found incomplete JSON array, attempting to complete it...")
            objects = []
            obj_start = -1
            brace_count = 0
            for i, char in enumerate(text[start_idx + 1 :], start=start_idx + 1):
                if char == "{":
                    if brace_count == 0:
                        obj_start = i
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0 and obj_start != -1:
                        objects.append(text[obj_start : i + 1])

            if objects:
                reconstructed_json = "[\n" + ",\n".join(objects) + "\n]"
                try:
                    return json.loads(reconstructed_json)
                except json.JSONDecodeError:
                    print("Couldn't parse reconstructed JSON array. Attempting repairs...")
                    fixed_json = re.sub(r"(\s*)(\w+)(\s*):(\s*)", r'\1"\2"\3:\4', reconstructed_json)
                    fixed_json = re.sub(r",(\s*[\]}])", r"\1", fixed_json)
                    try:
                        return json.loads(fixed_json)
                    except json.JSONDecodeError:
                        print("Could not fix JSON format issues in reconstructed array")

        print("No complete JSON array could be extracted")
        return None

