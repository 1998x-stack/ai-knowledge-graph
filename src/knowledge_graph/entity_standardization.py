"""Entity standardization and relationship inference for knowledge graphs."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Set

from src.knowledge_graph.llm import LLMService, extract_json_from_text
from src.knowledge_graph.prompts import (
    ENTITY_RESOLUTION_PROMPT,
    RELATIONSHIP_INFERENCE_PROMPT,
    WITHIN_COMMUNITY_INFERENCE_PROMPT,
)

def limit_predicate_length(predicate: str, max_words: int = 3) -> str:
    """
    Enforce a maximum word limit on predicates.
    
    Args:
        predicate: The original predicate string
        max_words: Maximum number of words allowed (default: 3)
        
    Returns:
        Shortened predicate with no more than max_words
    """
    words = predicate.split()
    if len(words) <= max_words:
        return predicate
    
    # If too long, use only the first max_words words
    shortened = ' '.join(words[:max_words])
    
    # Remove trailing prepositions or articles if they're the last word
    stop_words = {'a', 'an', 'the', 'of', 'with', 'by', 'to', 'from', 'in', 'on', 'for'}
    last_word = shortened.split()[-1].lower()
    if last_word in stop_words and len(words) > 1:
        shortened = ' '.join(shortened.split()[:-1])
    
    return shortened

def standardize_entities(
    triples: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    llm_service: LLMService | None = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Standardize entity names across all triples."""
    if not triples:
        return triples
    
    print("Standardizing entity names across all triples...")
    
    # Validate input triples to ensure they have the required fields
    valid_triples: List[Dict[str, Any]] = []
    invalid_count = 0
    
    for triple in triples:
        if isinstance(triple, dict) and "subject" in triple and "predicate" in triple and "object" in triple:
            valid_triples.append(triple)
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"Warning: Filtered out {invalid_count} invalid triples missing required fields")
    
    if not valid_triples:
        print("Error: No valid triples found for entity standardization")
        return []
    
    # 1. Extract all unique entities
    all_entities = set()
    for triple in valid_triples:
        all_entities.add(triple["subject"].lower())
        all_entities.add(triple["object"].lower())
    
    # 2. Group similar entities - first by exact match after lowercasing and removing stopwords
    standardized_entities = {}
    entity_groups = defaultdict(list)
    
    # Helper function to normalize text for comparison
    def normalize_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove common stopwords that might appear in entity names
        stopwords = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "with", "by", "as"}
        words = [word for word in re.findall(r'\b\w+\b', text) if word not in stopwords]
        return " ".join(words)
    
    # Process entities in order of complexity (longer entities first)
    sorted_entities = sorted(all_entities, key=lambda x: (-len(x), x))
    
    # First pass: Standard normalization
    for entity in sorted_entities:
        normalized = normalize_text(entity)
        if normalized:  # Skip empty strings
            entity_groups[normalized].append(entity)
    
    # 3. For each group, choose the most representative name
    for group_key, variants in entity_groups.items():
        if len(variants) == 1:
            # Only one variant, use it directly
            standardized_entities[variants[0]] = variants[0]
        else:
            # Multiple variants, choose the most common or the shortest one as standard
            # Sort by frequency in triples, then by length (shorter is better)
            variant_counts = defaultdict(int)
            for triple in valid_triples:
                for variant in variants:
                    if triple["subject"].lower() == variant:
                        variant_counts[variant] += 1
                    if triple["object"].lower() == variant:
                        variant_counts[variant] += 1
            
            # Choose the most common variant as the standard form
            standard_form = sorted(variants, key=lambda x: (-variant_counts[x], len(x)))[0]
            for variant in variants:
                standardized_entities[variant] = standard_form
    
    # 4. Second pass: check for root word relationships
    # This handles cases like "capitalism" and "capitalist decay"
    additional_standardizations = {}
    
    # Get all standardized entity names (after first pass)
    standard_forms = set(standardized_entities.values())
    sorted_standards = sorted(standard_forms, key=len)
    
    for i, entity1 in enumerate(sorted_standards):
        e1_words = set(entity1.split())
        
        for entity2 in sorted_standards[i+1:]:
            if entity1 == entity2:
                continue
                
            # Check if one entity is a subset of the other
            e2_words = set(entity2.split())
            
            # If one entity contains all words from the other
            if e1_words.issubset(e2_words) and len(e1_words) > 0:
                # The shorter one is likely the more general concept
                additional_standardizations[entity2] = entity1
            elif e2_words.issubset(e1_words) and len(e2_words) > 0:
                additional_standardizations[entity1] = entity2
            else:
                # Check for stemming/root similarities
                stems1 = {word[:4] for word in e1_words if len(word) > 4}
                stems2 = {word[:4] for word in e2_words if len(word) > 4}
                
                shared_stems = stems1.intersection(stems2)
                
                if shared_stems and (len(shared_stems) / max(len(stems1), len(stems2))) > 0.5:
                    # Use the shorter entity as the standard
                    if len(entity1) <= len(entity2):
                        additional_standardizations[entity2] = entity1
                    else:
                        additional_standardizations[entity1] = entity2
    
    # Apply additional standardizations
    for entity, standard in additional_standardizations.items():
        standardized_entities[entity] = standard
    
    # 5. Apply standardization to all triples
    standardized_triples: List[Dict[str, Any]] = []
    for triple in valid_triples:
        subj_lower = triple["subject"].lower()
        obj_lower = triple["object"].lower()
        
        standardized_triple = {
            "subject": standardized_entities.get(subj_lower, triple["subject"]),
            "predicate": limit_predicate_length(triple["predicate"]),
            "object": standardized_entities.get(obj_lower, triple["object"]),
            "chunk": triple.get("chunk", 0)
        }
        standardized_triples.append(standardized_triple)
    
    # 6. Optional: Use LLM to help with entity resolution for ambiguous cases
    if config.get("standardization", {}).get("use_llm_for_entities", False):
        if llm_service is None:
            print("LLM service unavailable; skipping LLM-based entity resolution")
        else:
            standardized_triples = _resolve_entities_with_llm(
                standardized_triples,
                llm_service,
                debug=debug,
            )
    
    # 7. Filter out self-referencing triples
    filtered_triples = [triple for triple in standardized_triples if triple["subject"] != triple["object"]]
    if len(filtered_triples) < len(standardized_triples):
        print(f"Removed {len(standardized_triples) - len(filtered_triples)} self-referencing triples")
    
    print(f"Standardized {len(all_entities)} entities into {len(set(standardized_entities.values()))} standard forms")
    return filtered_triples

def infer_relationships(
    triples: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    llm_service: LLMService | None = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Infer additional relationships between entities to reduce isolated communities."""

    if not triples or len(triples) < 2:
        return list(triples)
    
    print("Inferring additional relationships between entities...")
    
    # Validate input triples to ensure they have the required fields
    valid_triples: List[Dict[str, Any]] = []
    invalid_count = 0
    
    for triple in triples:
        if isinstance(triple, dict) and "subject" in triple and "predicate" in triple and "object" in triple:
            valid_triples.append(triple)
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"Warning: Filtered out {invalid_count} invalid triples missing required fields")
    
    if not valid_triples:
        print("Error: No valid triples found for relationship inference")
        return []
    
    # Create a graph representation for easier traversal
    graph = defaultdict(set)
    all_entities = set()
    for triple in valid_triples:
        subj = triple["subject"]
        obj = triple["object"]
        graph[subj].add(obj)
        all_entities.add(subj)
        all_entities.add(obj)
    
    # Find disconnected communities
    communities = _identify_communities(graph)
    print(f"Identified {len(communities)} disconnected communities in the graph")
    
    new_triples: List[Dict[str, Any]] = []
    
    # Use LLM to infer relationships between isolated communities if configured
    if config.get("inference", {}).get("use_llm_for_inference", True):
        if llm_service is None:
            print("LLM service unavailable; skipping LLM-based relationship inference")
        else:
            community_triples = _infer_relationships_with_llm(
                valid_triples,
                communities,
                llm_service,
                debug=debug,
            )
            if community_triples:
                new_triples.extend(community_triples)

            within_community_triples = _infer_within_community_relationships(
                valid_triples,
                communities,
                llm_service,
                debug=debug,
            )
            if within_community_triples:
                new_triples.extend(within_community_triples)
    
    # Apply transitive inference rules
    transitive_triples = _apply_transitive_inference(valid_triples, graph)
    if transitive_triples:
        new_triples.extend(transitive_triples)
    
    # Infer relationships based on lexical similarity
    lexical_triples = _infer_relationships_by_lexical_similarity(all_entities, valid_triples)
    if lexical_triples:
        new_triples.extend(lexical_triples)
    
    # Add new triples to the original set
    if new_triples:
        valid_triples.extend(new_triples)
    
    # De-duplicate triples
    unique_triples = _deduplicate_triples(valid_triples)
    
    # Final pass: ensure all predicates follow the 3-word limit
    for triple in unique_triples:
        triple["predicate"] = limit_predicate_length(triple["predicate"])
    
    # Filter out self-referencing triples
    filtered_triples = [triple for triple in unique_triples if triple["subject"] != triple["object"]]
    if len(filtered_triples) < len(unique_triples):
        print(f"Removed {len(unique_triples) - len(filtered_triples)} self-referencing triples")
    
    print(f"Added {len(filtered_triples) - len(triples)} inferred relationships")
    return filtered_triples

def _identify_communities(graph):
    """
    Identify disconnected communities in the graph.
    
    Args:
        graph: Dictionary representing the graph structure
        
    Returns:
        List of sets, where each set contains nodes in a community
    """
    # Get all nodes
    all_nodes = set(graph.keys()).union(*[graph[node] for node in graph])
    
    # Track visited nodes
    visited = set()
    communities = []
    
    # Depth-first search to find connected components
    def dfs(node, community):
        visited.add(node)
        community.add(node)
        
        # Visit outgoing edges
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, community)
        
        # Visit incoming edges (we need to check all nodes)
        for source, targets in graph.items():
            if node in targets and source not in visited:
                dfs(source, community)
    
    # Find all communities
    for node in all_nodes:
        if node not in visited:
            community = set()
            dfs(node, community)
            communities.append(community)
    
    return communities

def _apply_transitive_inference(triples, graph):
    """
    Apply transitive inference to find new relationships.
    
    Args:
        triples: List of triple dictionaries
        graph: Dictionary representing the graph structure
        
    Returns:
        List of new inferred triples
    """
    new_triples = []
    
    # Predicates by subject-object pairs
    predicates = {}
    for triple in triples:
        key = (triple["subject"], triple["object"])
        predicates[key] = triple["predicate"]
    
    # Find transitive relationships: A -> B -> C implies A -> C
    for subj in graph:
        for mid in graph[subj]:
            for obj in graph.get(mid, []):
                # Only consider paths where A->B->C and A!=C
                if subj != obj and (subj, obj) not in predicates:
                    # Create a new predicate combining the two relationships
                    pred1 = predicates.get((subj, mid), "relates to")
                    pred2 = predicates.get((mid, obj), "relates to")
                    
                    # Generate a new predicate based on the transitive relationship
                    new_pred = f"indirectly {pred1}" if pred1 == pred2 else f"{pred1} via {mid}"
                    
                    # Add the new transitive relationship
                    new_triples.append({
                        "subject": subj,
                        "predicate": limit_predicate_length(new_pred),
                        "object": obj,
                        "inferred": True  # Mark as inferred
                    })
    
    return new_triples

def _deduplicate_triples(triples):
    """
    Remove duplicate triples, keeping the original (non-inferred) ones.
    
    Args:
        triples: List of triple dictionaries
        
    Returns:
        List of unique triples
    """
    # Use tuple of (subject, predicate, object) as key
    unique_triples = {}
    
    for triple in triples:
        key = (triple["subject"], triple["predicate"], triple["object"])
        # Keep original triples (not inferred) when duplicates exist
        if key not in unique_triples or not triple.get("inferred", False):
            unique_triples[key] = triple
    
    return list(unique_triples.values())

def _resolve_entities_with_llm(
    triples: List[Dict[str, Any]],
    llm_service: LLMService,
    *,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Use an LLM chain to standardize ambiguous entity references."""

    all_entities: Set[str] = set()
    for triple in triples:
        all_entities.add(triple["subject"])
        all_entities.add(triple["object"])

    if len(all_entities) > 100:
        entity_counts: Dict[str, int] = defaultdict(int)
        for triple in triples:
            entity_counts[triple["subject"]] += 1
            entity_counts[triple["object"]] += 1
        top_entities = sorted(entity_counts.items(), key=lambda item: -item[1])[:100]
        all_entities = {entity for entity, _ in top_entities}

    if not all_entities:
        return triples

    entity_list = "\n".join(sorted(all_entities))

    try:
        response = llm_service.invoke(
            ENTITY_RESOLUTION_PROMPT.name,
            {"entity_list": entity_list},
            debug=debug,
        )
    except Exception as exc:
        print(f"Error in LLM-based entity resolution: {exc}")
        return triples

    entity_mapping = extract_json_from_text(response)

    if not entity_mapping or not isinstance(entity_mapping, dict):
        print("Could not extract valid entity mapping from LLM response")
        return triples

    entity_to_standard: Dict[str, str] = {}
    for standard, variants in entity_mapping.items():
        for variant in variants:
            entity_to_standard[variant] = standard
        entity_to_standard[standard] = standard

    for triple in triples:
        triple["subject"] = entity_to_standard.get(triple["subject"], triple["subject"])
        triple["object"] = entity_to_standard.get(triple["object"], triple["object"])

    print(f"Applied LLM-based entity standardization for {len(entity_mapping)} entity groups")
    return triples

def _infer_relationships_with_llm(
    triples: Sequence[Dict[str, Any]],
    communities: Sequence[Set[str]],
    llm_service: LLMService,
    *,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Use an LLM to infer relationships between disconnected communities."""

    if len(communities) <= 1:
        print("Only one community found, skipping LLM-based relationship inference")
        return []

    large_communities = sorted(communities, key=len, reverse=True)[:5]
    new_triples: List[Dict[str, Any]] = []

    for i, comm1 in enumerate(large_communities):
        for j, comm2 in enumerate(large_communities):
            if i >= j:
                continue

            rep1 = list(comm1)[: min(5, len(comm1))]
            rep2 = list(comm2)[: min(5, len(comm2))]
            
            context_triples = [
                triple
                for triple in triples
                if triple["subject"] in rep1
                or triple["subject"] in rep2
                or triple["object"] in rep1
                or triple["object"] in rep2
            ][:20]

            triples_text = "\n".join(
                f"{t['subject']} {t['predicate']} {t['object']}" for t in context_triples
            )

            entities1 = ", ".join(rep1)
            entities2 = ", ".join(rep2)

            try:
                response = llm_service.invoke(
                    RELATIONSHIP_INFERENCE_PROMPT.name,
                    {
                        "entities1": entities1,
                        "entities2": entities2,
                        "triples_text": triples_text,
                    },
                    debug=debug,
                )
            except Exception as exc:
                print(f"Error in LLM-based relationship inference: {exc}")
                continue

            inferred_triples = extract_json_from_text(response)

            if not inferred_triples or not isinstance(inferred_triples, list):
                print("Could not extract valid inferred relationships from LLM response")
                continue

            for triple in inferred_triples:
                if {"subject", "predicate", "object"}.issubset(triple):
                    if triple["subject"] == triple["object"]:
                        continue
                    triple["inferred"] = True
                    triple["predicate"] = limit_predicate_length(triple["predicate"])
                    new_triples.append(triple)

            print(f"Inferred {len(inferred_triples)} new relationships between communities")

    return new_triples

def _infer_within_community_relationships(
    triples: Sequence[Dict[str, Any]],
    communities: Sequence[Set[str]],
    llm_service: LLMService,
    *,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Use an LLM to infer relationships within a community."""

    new_triples: List[Dict[str, Any]] = []
    
    # Process larger communities
    for community in sorted(communities, key=len, reverse=True)[:3]:
        # Skip small communities
        if len(community) < 5:
            continue
            
        # Get all entities in this community
        community_entities = list(community)
        
        # Create an adjacency matrix to identify disconnected entity pairs
        connections = {(a, b): False for a in community_entities for b in community_entities if a != b}
        
        # Mark existing connections
        for triple in triples:
            if triple["subject"] in community_entities and triple["object"] in community_entities:
                connections[(triple["subject"], triple["object"])] = True
        
        # Find disconnected pairs that might be semantically related
        disconnected_pairs = []
        for (a, b), connected in connections.items():
            if not connected and a != b:  # Ensure a and b are different entities
                # Check for potential semantic relationship (e.g., shared words)
                a_words = set(a.lower().split())
                b_words = set(b.lower().split())
                shared_words = a_words.intersection(b_words)
                
                # If they share words or one is contained in the other, they might be related
                if shared_words or a.lower() in b.lower() or b.lower() in a.lower():
                    disconnected_pairs.append((a, b))
        
        # Limit to the most promising pairs
        disconnected_pairs = disconnected_pairs[:10]
        
        if not disconnected_pairs:
            continue
            
        # Get relevant context
        context_triples = []
        entities_of_interest = set()
        for a, b in disconnected_pairs:
            entities_of_interest.add(a)
            entities_of_interest.add(b)
            
        for triple in triples:
            if triple["subject"] in entities_of_interest or triple["object"] in entities_of_interest:
                context_triples.append(triple)
        
        # Limit context size
        if len(context_triples) > 20:
            context_triples = context_triples[:20]
            
        # Convert triples to text for prompt
        triples_text = "\n".join([
            f"{t['subject']} {t['predicate']} {t['object']}"
            for t in context_triples
        ])
        
        # Create pairs text
        pairs_text = "\n".join([f"{a} and {b}" for a, b in disconnected_pairs])
        
        try:
            response = llm_service.invoke(
                WITHIN_COMMUNITY_INFERENCE_PROMPT.name,
                {
                    "pairs_text": pairs_text,
                    "triples_text": triples_text,
                },
                debug=debug,
            )
        except Exception as exc:
            print(f"Error in LLM-based relationship inference within communities: {exc}")
            continue

        inferred_triples = extract_json_from_text(response)

        if not inferred_triples or not isinstance(inferred_triples, list):
            print("Could not extract valid inferred relationships from LLM response")
            continue

        for triple in inferred_triples:
            if {"subject", "predicate", "object"}.issubset(triple):
                if triple["subject"] == triple["object"]:
                    continue
                triple["inferred"] = True
                triple["predicate"] = limit_predicate_length(triple["predicate"])
                new_triples.append(triple)

        print(f"Inferred {len(inferred_triples)} new relationships within communities")

    return new_triples

def _infer_relationships_by_lexical_similarity(entities, triples):
    """
    Infer relationships between entities based on lexical similarity.
    This can help connect entities like "capitalism" and "capitalist decay".
    
    Args:
        entities: Set of all entities
        triples: List of existing triples
        
    Returns:
        List of new inferred triples
    """
    new_triples = []
    processed_pairs = set()
    
    # Create a dictionary to track existing relationships
    existing_relationships = set()
    for triple in triples:
        existing_relationships.add((triple["subject"], triple["object"]))
    
    # Check for lexical similarity between entities
    entities_list = list(entities)
    for i, entity1 in enumerate(entities_list):
        for entity2 in entities_list[i+1:]:
            # Skip if already connected
            if (entity1, entity2) in existing_relationships or (entity2, entity1) in existing_relationships:
                continue
                
            # Skip if already processed this pair
            if (entity1, entity2) in processed_pairs or (entity2, entity1) in processed_pairs:
                continue
                
            # Skip if the entities are the same (prevent self-reference)
            if entity1 == entity2:
                continue
                
            processed_pairs.add((entity1, entity2))
            
            # Check for containment or shared roots
            e1_lower = entity1.lower()
            e2_lower = entity2.lower()
            
            # Simple word overlap check
            e1_words = set(e1_lower.split())
            e2_words = set(e2_lower.split())
            shared_words = e1_words.intersection(e2_words)
            
            if shared_words:
                # Create relationships based on shared words
                main_shared = max(shared_words, key=len)
                
                if len(main_shared) >= 4:  # Only consider significant shared words
                    if e1_lower.startswith(main_shared) and not e2_lower.startswith(main_shared):
                        new_triples.append({
                            "subject": entity2,
                            "predicate": "relates to",
                            "object": entity1,
                            "inferred": True
                        })
                    elif e2_lower.startswith(main_shared) and not e1_lower.startswith(main_shared):
                        new_triples.append({
                            "subject": entity1,
                            "predicate": "relates to",
                            "object": entity2,
                            "inferred": True
                        })
                    else:
                        new_triples.append({
                            "subject": entity1,
                            "predicate": "related to",
                            "object": entity2,
                            "inferred": True
                        })
            
            # Check if one entity contains the other
            elif e1_lower in e2_lower:
                new_triples.append({
                    "subject": entity2,
                    "predicate": "is type of",
                    "object": entity1,
                    "inferred": True
                })
            elif e2_lower in e1_lower:
                new_triples.append({
                    "subject": entity1,
                    "predicate": "is type of",
                    "object": entity2,
                    "inferred": True
                })
    
    print(f"Inferred {len(new_triples)} relationships based on lexical similarity")
    return new_triples 