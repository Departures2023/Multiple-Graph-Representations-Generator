from __future__ import annotations

from typing import List, Optional, Tuple, Union

import networkx as nx
from PIL import Image

from description_to_image import generate_graph_image
from src.graph_title import generate_title, lookup_graph

ImageType = Image.Image


class Graph:
    """
    A graph abstraction that supports three interchangeable representations:
      - description: canonical representation (list of edges, optionally weighted)
      - image: rendered diagram or visual object
      - title: human-readable name for the graph
    
    Edge formats supported:
      - Unweighted: (node1, node2)
      - Weighted: (node1, node2, weight)
    
    Nodes can be integers or strings (for labeled graphs).
    """

    def __init__(
        self,
        description: Optional[List[Union[Tuple, Tuple[int, int], Tuple[str, str], Tuple[int, int, float], Tuple[str, str, float]]]] = None,
        image: Optional[ImageType] = None,
        title: Optional[str] = None,
    ) -> None:

        if description is None and image is None and title is None:
            raise ValueError("At least one representation must be provided.")

        self.description = description
        self.image = image
        self.title = title
        self._complete_representations()

    # -----------------------------------------------------------------------
    # MAIN RESOLUTION LOGIC
    # -----------------------------------------------------------------------

    def _complete_representations(self) -> None:
        """
        Iteratively fill missing representations using whatever is available.
        """

        while True:
            made_progress = False

            # Try to derive description if missing
            if self.description is None:
                if self.title:
                    desc = self._title_to_description(self.title)
                    if desc:
                        self.description = desc
                        made_progress = True
                if self.description is None and self.image is not None:
                    desc = self._image_to_description(self.image)
                    if desc:
                        self.description = desc
                        made_progress = True

            # If we have description, derive title/image as needed
            if self.description is not None:
                if self.title is None:
                    try:
                        self.title = self._description_to_title(self.description)
                        made_progress = True
                    except Exception:
                        pass
                if self.image is None:
                    try:
                        self.image = self._description_to_image(self.description)
                        made_progress = True
                    except Exception:
                        pass

            if not made_progress:
                break

    # -----------------------------------------------------------------------
    # CONVERSION METHODS
    # These should be implemented.
    # -----------------------------------------------------------------------

    @staticmethod
    def _description_to_image(description: List) -> ImageType:
        # Extract edges (remove weights if present)
        edges = []
        for edge in description:
            if len(edge) >= 2:
                edges.append((edge[0], edge[1]))
        return generate_graph_image(edges)

    @staticmethod
    def _description_to_title(description: List) -> str:
        G = nx.Graph()
        # Handle both weighted and unweighted edges
        for edge in description:
            if len(edge) == 3:
                # Weighted edge
                G.add_edge(edge[0], edge[1], weight=edge[2])
            else:
                # Unweighted edge
                G.add_edge(edge[0], edge[1])
        return generate_title(G)

    @staticmethod
    def _title_to_description(title: str) -> Optional[List[Tuple[int, int]]]:
        """
        Reverse lookup from canonical title → edge list.
        """
        G = lookup_graph(title)
        if G is None:
            return None
        return list(G.edges())

    @staticmethod
    def _image_to_description(image: ImageType) -> Optional[List]:
        """
        Convert a graph image to a list of edges (description) with labels and weights.
        
        Uses the ImprovedGraphDetector to detect nodes, edges, labels, and weights.
        Returns a list of edge tuples:
        - Weighted: (source_label, target_label, weight)
        - Unweighted: (source_label, target_label)
        
        Args:
            image: PIL Image containing a graph diagram
            
        Returns:
            List of edge tuples with labels and optional weights, or None if detection fails
        """
        try:
            from image_to_description.improved_detector import ImprovedGraphDetector
            
            # Create detector with OCR enabled
            detector = ImprovedGraphDetector(image, use_ocr=True)
            
            # Detect nodes and edges with optimized parameters
            nodes = detector.detect_nodes(min_radius=20, max_radius=40)
            if not nodes:
                return None
            
            edges = detector.detect_edges(detect_arrows=True, edge_min_pixels=2, node_proximity=40)
            if not edges:
                return None
            
            # Create mapping from node IDs to labels
            node_id_to_label = {}
            for node in nodes:
                if node['text']:
                    node_id_to_label[node['id']] = node['text']
                else:
                    node_id_to_label[node['id']] = str(node['id'])
            
            # Build edge list using labels and weights
            edge_list = []
            for e in edges:
                src_label = node_id_to_label[e['source']]
                tgt_label = node_id_to_label[e['target']]
                
                if e.get('weight') is not None:
                    # Weighted edge
                    edge_list.append((src_label, tgt_label, e['weight']))
                else:
                    # Unweighted edge
                    edge_list.append((src_label, tgt_label))
            
            return edge_list
            
        except Exception:
            # If detection fails, return None
            return None

    def __repr__(self) -> str:
        return (
            f"Graph(description={self.description}, " f"title={self.title}, " f"image={self.image})"
        )
