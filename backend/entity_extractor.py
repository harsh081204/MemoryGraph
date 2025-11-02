"""
Enhanced entity extraction module using spaCy NLP with relationship detection.
"""
import spacy
from typing import List, Dict
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model (with error handling)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    raise


def extract_entities(text: str) -> List[Dict]:
    """
    Enhanced entity extraction with better relationship detection.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of entity dictionaries with name and relations
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for entity extraction")
        return [{"name": "Unknown Entity", "relations": []}]
    
    doc = nlp(text)
    entity_dict = {}
    
    # Extract unique entities with enhanced processing
    for ent in doc.ents:
        entity = ent.text.strip()
        if entity and entity not in entity_dict:
            entity_dict[entity] = {
                "name": entity, 
                "relations": [],
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
    
    # Enhanced relationship detection
    for sent in doc.sents:
        sent_ents = [ent for ent in sent.ents if ent.text.strip()]
        
        if len(sent_ents) > 1:
            # Extract relationships using dependency parsing
            relations = _extract_relationships(sent, sent_ents)
            
            for source, target, rel_type in relations:
                if source in entity_dict:
                    # Avoid duplicate relations
                    existing_targets = {rel["target"] for rel in entity_dict[source]["relations"]}
                    if target not in existing_targets:
                        entity_dict[source]["relations"].append({
                            "target": target,
                            "type": rel_type
                        })
    
    entities = list(entity_dict.values())
    
    # Fallback if no entities found
    if not entities:
        logger.info("No entities found in text")
        entities.append({"name": "Unknown Entity", "relations": []})
    
    logger.info(f"Extracted {len(entities)} entities with enhanced relationships")
    return entities


def _extract_relationships(sent, entities):
    """
    Extract relationships between entities using dependency parsing.
    
    Args:
        sent: spaCy sentence object
        entities: List of entities in the sentence
        
    Returns:
        List of (source, target, relationship_type) tuples
    """
    relationships = []
    
    for i, ent1 in enumerate(entities):
        for j, ent2 in enumerate(entities):
            if i >= j:  # Avoid duplicates and self-relations
                continue
                
            # Check for direct syntactic relationships
            rel_type = _find_relationship_type(sent, ent1, ent2)
            if rel_type:
                relationships.append((ent1.text.strip(), ent2.text.strip(), rel_type))
    
    return relationships


def _find_relationship_type(sent, ent1, ent2):
    """
    Find the relationship type between two entities using dependency parsing.
    
    Args:
        sent: spaCy sentence object
        ent1, ent2: Entity objects
        
    Returns:
        Relationship type string or None
    """
    # Get the root tokens of the entities
    ent1_root = ent1.root
    ent2_root = ent2.root
    
    # Check for direct dependency relationships
    if ent1_root.head == ent2_root:
        return _classify_dependency(ent1_root.dep_, "forward")
    elif ent2_root.head == ent1_root:
        return _classify_dependency(ent2_root.dep_, "backward")
    
    # Check for common ancestors
    ent1_ancestors = [token for token in ent1_root.ancestors]
    ent2_ancestors = [token for token in ent2_root.ancestors]
    
    common_ancestors = set(ent1_ancestors) & set(ent2_ancestors)
    if common_ancestors:
        # Entities share a common ancestor - they're related
        return "RELATED_TO"
    
    # Check for proximity-based relationships
    if abs(ent1.start - ent2.end) <= 10 or abs(ent2.start - ent1.end) <= 10:
        return "CO_OCCURS_WITH"
    
    return None


def _classify_dependency(dep, direction):
    """
    Classify dependency relationship into semantic types.
    
    Args:
        dep: Dependency label
        direction: "forward" or "backward"
        
    Returns:
        Semantic relationship type
    """
    dep_mapping = {
        "nsubj": "SUBJECT_OF",
        "dobj": "OBJECT_OF", 
        "pobj": "OBJECT_OF",
        "amod": "DESCRIBES",
        "compound": "PART_OF",
        "poss": "OWNED_BY",
        "prep": "RELATED_TO",
        "conj": "SIMILAR_TO",
        "appos": "ALSO_KNOWN_AS"
    }
    
    return dep_mapping.get(dep, "RELATED_TO")