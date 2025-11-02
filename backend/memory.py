"""
Knowledge graph memory module for storing entities and relationships.
"""
from typing import List, Dict, Optional, Tuple
import pickle
import networkx as nx
import logging
from pathlib import Path
import numpy as np
from collections import defaultdict
import time
from cache import cached

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryGraph:
    """Persistent knowledge graph for storing entities and their relationships."""
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize memory graph.
        
        Args:
            persist_path: Path to pickle file for persistence (optional)
        """
        self.graph = nx.DiGraph()
        self.persist_path = persist_path
        self._unsaved_changes = False

        if persist_path:
            self._load_graph()
    
    def _load_graph(self):
        """Load graph from disk."""
        try:
            path = Path(self.persist_path)
            if path.exists():
                with open(self.persist_path, "rb") as f:
                    self.graph = pickle.load(f)
                logger.info(f"Loaded graph with {len(self.graph.nodes())} nodes from {self.persist_path}")
            else:
                logger.info("No existing graph found. Starting fresh.")
        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(f"Corrupted graph file: {e}. Starting fresh.")
            self.graph = nx.DiGraph()
        except Exception as e:
            logger.error(f"Unexpected error loading graph: {e}")
            self.graph = nx.DiGraph()
    
    def add_entities(self, entities: List[Dict]):
        """
        Add entities and their relations to the graph.
        
        Args:
            entities: List of entity dictionaries
        """
        if not entities:
            logger.warning("No entities to add")
            return
            
        for entity in entities:
            node_name = entity.get("name", "").strip()
            if not node_name or node_name == "Unknown Entity":
                continue
                
            self.graph.add_node(node_name)
            self._unsaved_changes = True

            relations = entity.get("relations", [])
            for rel in relations:
                target = rel.get("target", "").strip()
                if not target:
                    continue
                    
                rel_type = rel.get("type", "RELATED_TO")
                self.graph.add_node(target)
                
                # Update edge if exists, add if not
                if self.graph.has_edge(node_name, target):
                    self.graph.edges[node_name, target]["type"] = rel_type
                else:
                    self.graph.add_edge(node_name, target, type=rel_type)
                    
                self._unsaved_changes = True
        
        logger.info(f"Graph now has {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")

    def get_context(self, entities: List[Dict], top_k: int = 5) -> str:
        """
        Retrieve context about entities from the graph using multiple algorithms.
        
        Args:
            entities: List of entities to get context for
            top_k: Maximum number of relations per entity
            
        Returns:
            Formatted context string
        """
        context_sentences = set()
        
        for entity in entities:
            node_name = entity.get("name", "").strip()
            if not node_name or node_name not in self.graph:
                continue
            
            # Get direct relations
            context_sentences.update(self._get_direct_relations(node_name, top_k))
            
            # Get 2-hop neighbors for richer context
            context_sentences.update(self._get_extended_context(node_name, top_k))
        
        if not context_sentences:
            return "No prior context available."
            
        return "\n".join(sorted(context_sentences))
    
    def _get_direct_relations(self, node_name: str, top_k: int) -> set:
        """Get direct relations for a node."""
        relations = set()
        
        # Get outgoing relations
        for neighbor in list(self.graph.successors(node_name))[:top_k]:
            rel_type = self.graph.edges[node_name, neighbor].get("type", "RELATED_TO")
            relations.add(f"{node_name} {rel_type} {neighbor}")

        # Get incoming relations
        for pred in list(self.graph.predecessors(node_name))[:top_k]:
            rel_type = self.graph.edges[pred, node_name].get("type", "RELATED_TO")
            relations.add(f"{pred} {rel_type} {node_name}")
        
        return relations
    
    def _get_extended_context(self, node_name: str, top_k: int) -> set:
        """Get extended context using graph traversal algorithms."""
        relations = set()
        
        try:
            # Use BFS to find 2-hop neighbors
            visited = set()
            queue = [(node_name, 0)]  # (node, depth)
            
            while queue and len(relations) < top_k * 2:
                current_node, depth = queue.pop(0)
                
                if current_node in visited or depth > 2:
                    continue
                    
                visited.add(current_node)
                
                if depth == 1:  # 1-hop neighbors
                    for neighbor in self.graph.neighbors(current_node):
                        if neighbor != node_name:
                            rel_type = self.graph.edges[current_node, neighbor].get("type", "RELATED_TO")
                            relations.add(f"{current_node} {rel_type} {neighbor}")
                
                if depth < 2:  # Continue to 2-hop
                    for neighbor in self.graph.neighbors(current_node):
                        if neighbor not in visited:
                            queue.append((neighbor, depth + 1))
                            
        except Exception as e:
            logger.warning(f"Error in extended context retrieval: {e}")
        
        return relations
    
    @cached(ttl=600, key_prefix="centrality_")  # Cache for 10 minutes
    def find_central_entities(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most central entities in the graph using various centrality measures.
        
        Args:
            top_k: Number of top entities to return
            
        Returns:
            List of (entity, centrality_score) tuples
        """
        if len(self.graph.nodes()) < 2:
            return []
        
        try:
            # Calculate multiple centrality measures
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            closeness_centrality = nx.closeness_centrality(self.graph)
            
            # Combine centrality measures (weighted average)
            combined_centrality = {}
            for node in self.graph.nodes():
                combined_centrality[node] = (
                    degree_centrality.get(node, 0) * 0.4 +
                    betweenness_centrality.get(node, 0) * 0.3 +
                    closeness_centrality.get(node, 0) * 0.3
                )
            
            # Sort by combined centrality
            sorted_entities = sorted(
                combined_centrality.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return sorted_entities[:top_k]
            
        except Exception as e:
            logger.error(f"Error calculating centrality: {e}")
            return []
    
    def find_related_entities(self, entity_name: str, max_distance: int = 2) -> List[str]:
        """
        Find entities related to a given entity within a certain distance.
        
        Args:
            entity_name: Name of the entity to find relations for
            max_distance: Maximum graph distance to consider
            
        Returns:
            List of related entity names
        """
        if entity_name not in self.graph:
            return []
        
        related = set()
        visited = set()
        queue = [(entity_name, 0)]
        
        while queue:
            current, distance = queue.pop(0)
            
            if current in visited or distance > max_distance:
                continue
                
            visited.add(current)
            if distance > 0:  # Don't include the entity itself
                related.add(current)
            
            if distance < max_distance:
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
        
        return list(related)
    
    def save_graph(self):
        """Persist the graph to disk if there are unsaved changes."""
        if not self.persist_path:
            logger.warning("No persist_path set. Graph not saved.")
            return False
        
        if not self._unsaved_changes:
            logger.debug("No changes to save")
            return True
            
        try:
            # Ensure directory exists
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.persist_path, "wb") as f:
                pickle.dump(self.graph, f)
            self._unsaved_changes = False
            logger.info(f"Graph saved to {self.persist_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            return False

    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self._unsaved_changes

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        return {
            "nodes": len(self.graph.nodes()),
            "edges": len(self.graph.edges()),
            "unsaved_changes": self._unsaved_changes
        }

    def export_graph(self, format: str = "json") -> Dict:
        """
        Export graph data in various formats.
        
        Args:
            format: Export format ("json", "gexf", "graphml")
            
        Returns:
            Exported graph data
        """
        if format == "json":
            return {
                "nodes": [
                    {
                        "id": node,
                        "data": self.graph.nodes[node]
                    }
                    for node in self.graph.nodes()
                ],
                "edges": [
                    {
                        "source": u,
                        "target": v,
                        "data": data
                    }
                    for u, v, data in self.graph.edges(data=True)
                ],
                "metadata": {
                    "export_time": time.time(),
                    "node_count": len(self.graph.nodes()),
                    "edge_count": len(self.graph.edges())
                }
            }
        elif format == "gexf":
            return nx.write_gexf(self.graph, "memory_graph.gexf")
        elif format == "graphml":
            return nx.write_graphml(self.graph, "memory_graph.graphml")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_graph(self, data: Dict) -> bool:
        """
        Import graph data from JSON format.
        
        Args:
            data: Graph data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear existing graph
            self.graph.clear()
            
            # Add nodes
            for node_data in data.get("nodes", []):
                node_id = node_data["id"]
                node_attrs = node_data.get("data", {})
                self.graph.add_node(node_id, **node_attrs)
            
            # Add edges
            for edge_data in data.get("edges", []):
                source = edge_data["source"]
                target = edge_data["target"]
                edge_attrs = edge_data.get("data", {})
                self.graph.add_edge(source, target, **edge_attrs)
            
            self._unsaved_changes = True
            logger.info(f"Imported graph with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import graph: {e}")
            return False
    
    def backup_graph(self, backup_path: str) -> bool:
        """
        Create a backup of the current graph.
        
        Args:
            backup_path: Path to save backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            backup_data = self.export_graph("json")
            with open(backup_path, "w") as f:
                import json
                json.dump(backup_data, f, indent=2)
            logger.info(f"Graph backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup graph: {e}")
            return False
    
    def restore_graph(self, backup_path: str) -> bool:
        """
        Restore graph from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(backup_path, "r") as f:
                import json
                backup_data = json.load(f)
            
            success = self.import_graph(backup_data)
            if success:
                logger.info(f"Graph restored from {backup_path}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to restore graph: {e}")
            return False

    def print_graph(self):
        """Print graph structure for debugging."""
        print(f"\n=== Memory Graph Stats ===")
        print(f"Nodes ({len(self.graph.nodes())}): {list(self.graph.nodes())[:10]}")
        if len(self.graph.nodes()) > 10:
            print(f"  ... and {len(self.graph.nodes()) - 10} more")
        
        edges = [(u, v, d['type']) for u, v, d in self.graph.edges(data=True)]
        print(f"Edges ({len(edges)}): {edges[:10]}")
        if len(edges) > 10:
            print(f"  ... and {len(edges) - 10} more")
        print("=" * 27 + "\n")