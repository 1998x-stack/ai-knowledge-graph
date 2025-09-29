"""
Knowledge Graph Generator and Visualizer.

A tool that takes text input and generates an interactive knowledge graph visualization.
"""

from src.knowledge_graph.visualization import visualize_knowledge_graph, sample_data_visualization
from src.knowledge_graph.llm import LLMService, extract_json_from_text, service_from_config
from src.knowledge_graph.config import load_config

__version__ = "0.1.0"
