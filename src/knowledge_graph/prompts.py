"""Centralized management of all prompts used by the knowledge graph system."""

from __future__ import annotations

from typing import Dict

from src.knowledge_graph.chain_factory import PromptTemplateConfig


TRIPLE_EXTRACTION_PROMPT = PromptTemplateConfig(
    name="triple_extraction",
    system_template=(
        "You are an advanced AI system specialized in knowledge extraction and knowledge graph generation.\n"
        "Your expertise includes identifying consistent entity references and meaningful relationships in text.\n"
        "CRITICAL INSTRUCTION: All relationships (predicates) MUST be no more than 3 words maximum. Ideally 1-2 words."
    ),
    user_template=(
        "Your task: Read the text below (delimited by triple backticks) and identify all Subject-Predicate-Object (S-P-O) "
        "relationships in each sentence. Then produce a single JSON array of objects, each representing one triple.\n\n"
        "Follow these rules carefully:\n\n"
        "- Entity Consistency: Use consistent names for entities throughout the document. For example, if \"John Smith\" is "
        "mentioned as \"John\", \"Mr. Smith\", and \"John Smith\" in different places, use a single consistent form "
        "(preferably the most complete one) in all triples.\n"
        "- Atomic Terms: Identify distinct key terms (e.g., objects, locations, organizations, acronyms, people, conditions, "
        "concepts, feelings). Avoid merging multiple ideas into one term (they should be as atomistic as possible).\n"
        "- Unified References: Replace any pronouns (e.g., he, she, it, they, etc.) with the actual referenced entity, if "
        "identifiable.\n"
        "- Pairwise Relationships: If multiple terms co-occur in the same sentence (or a short paragraph that makes them "
        "contextually related), create one triple for each pair that has a meaningful relationship.\n"
        "- CRITICAL INSTRUCTION: Predicates MUST be 1-3 words maximum. Never more than 3 words. Keep them extremely concise.\n"
        "- Ensure that all possible relationships are identified in the text and are captured in an S-P-O relation.\n"
        "- Standardize terminology: If the same concept appears with slight variations (e.g., \"artificial intelligence\" and "
        "\"AI\"), use the most common or canonical form consistently.\n"
        "- Make all the text of S-P-O text lower-case, even names of people and places.\n"
        "- If a person is mentioned by name, create a relation to their location, profession, and what they are known for if it "
        "fits the context.\n\n"
        "Important Considerations:\n\n"
        "- Aim for precision in entity naming - use specific forms that distinguish between similar but different entities.\n"
        "- Maximize connectedness by using identical entity names for the same concepts throughout the document.\n"
        "- Consider the entire context when identifying entity references.\n"
        "- ALL PREDICATES MUST BE 3 WORDS OR FEWER - this is a hard requirement.\n\n"
        "Output Requirements:\n\n"
        "- Do not include any text or commentary outside of the JSON.\n"
        "- Return only the JSON array, with each triple as an object containing \"subject\", \"predicate\", and \"object\".\n"
        "- Make sure the JSON is valid and properly formatted.\n\n"
        "Example of the desired output structure:\n\n"
        "[\n  {\n    \"subject\": \"term a\",\n    \"predicate\": \"relates to\",\n    \"object\": \"term b\"\n  },\n  {\n    \"subject\": \"term c\",\n    \"predicate\": \"uses\",\n    \"object\": \"term d\"\n  }\n]\n\n"
        "Important: Only output the JSON array (with the S-P-O objects) and nothing else.\n\n"
        "Text to analyze (between triple backticks):\n"
        "```{input_text}```"
    ),
)


ENTITY_RESOLUTION_PROMPT = PromptTemplateConfig(
    name="entity_resolution",
    system_template=(
        "You are an expert in entity resolution and knowledge representation.\n"
        "Your task is to standardize entity names from a knowledge graph to ensure consistency."
    ),
    user_template=(
        "Below is a list of entity names extracted from a knowledge graph.\n"
        "Some may refer to the same real-world entities but with different wording.\n\n"
        "Please identify groups of entities that refer to the same concept, and provide a standardized name for each group.\n"
        "Return your answer as a JSON object where the keys are the standardized names and the values are arrays of all variant "
        "names that should map to that standard name. Only include entities that have multiple variants or need standardization.\n\n"
        "Entity list:\n{entity_list}\n\n"
        "Format your response as valid JSON like this:\n"
        "{{\n  \"standardized name 1\": [\"variant 1\", \"variant 2\"],\n  \"standardized name 2\": [\"variant 3\", \"variant 4\", \"variant 5\"]\n}}"
    ),
)


RELATIONSHIP_INFERENCE_PROMPT = PromptTemplateConfig(
    name="relationship_inference",
    system_template=(
        "You are an expert in knowledge representation and inference.\n"
        "Your task is to infer plausible relationships between disconnected entities in a knowledge graph."
    ),
    user_template=(
        "I have a knowledge graph with two disconnected communities of entities.\n\n"
        "Community 1 entities: {entities1}\n"
        "Community 2 entities: {entities2}\n\n"
        "Here are some existing relationships involving these entities:\n{triples_text}\n\n"
        "Please infer 2-3 plausible relationships between entities from Community 1 and entities from Community 2.\n"
        "Return your answer as a JSON array of triples in the following format:\n\n"
        "[\n  {{\n    \"subject\": \"entity from community 1\",\n    \"predicate\": \"inferred relationship\",\n    \"object\": \"entity from community 2\"\n  }},\n  ...\n]\n\n"
        "Only include highly plausible relationships with clear predicates.\n"
        "IMPORTANT: The inferred relationships (predicates) MUST be no more than 3 words maximum. Preferably 1-2 words. Never "
        "more than 3.\n"
        "For predicates, use short phrases that clearly describe the relationship.\n"
        "IMPORTANT: Make sure the subject and object are different entities - avoid self-references."
    ),
)


WITHIN_COMMUNITY_INFERENCE_PROMPT = PromptTemplateConfig(
    name="within_community_inference",
    system_template=(
        "You are an expert in knowledge representation and inference.\n"
        "Your task is to infer plausible relationships between semantically related entities that are not yet connected in a "
        "knowledge graph."
    ),
    user_template=(
        "I have a knowledge graph with several entities that appear to be semantically related but are not directly connected.\n\n"
        "Here are some pairs of entities that might be related:\n{pairs_text}\n\n"
        "Here are some existing relationships involving these entities:\n{triples_text}\n\n"
        "Please infer plausible relationships between these disconnected pairs.\n"
        "Return your answer as a JSON array of triples in the following format:\n\n"
        "[\n  {{\n    \"subject\": \"entity1\",\n    \"predicate\": \"inferred relationship\",\n    \"object\": \"entity2\"\n  }},\n  ...\n]\n\n"
        "Only include highly plausible relationships with clear predicates.\n"
        "IMPORTANT: The inferred relationships (predicates) MUST be no more than 3 words maximum. Preferably 1-2 words. Never "
        "more than 3.\n"
        "IMPORTANT: Make sure that the subject and object are different entities - avoid self-references."
    ),
)


class PromptRegistry:
    """Provides access to the prompt templates used across the application."""

    def __init__(self) -> None:
        self._templates: Dict[str, PromptTemplateConfig] = {
            template.name: template
            for template in (
                TRIPLE_EXTRACTION_PROMPT,
                ENTITY_RESOLUTION_PROMPT,
                RELATIONSHIP_INFERENCE_PROMPT,
                WITHIN_COMMUNITY_INFERENCE_PROMPT,
            )
        }

    def get(self, name: str) -> PromptTemplateConfig:
        """Return the prompt template registered under the provided name."""

        try:
            return self._templates[name]
        except KeyError as exc:
            raise ValueError(f"Prompt '{name}' is not registered.") from exc

