from __future__ import annotations

from typing import List, Optional, Tuple

import networkx as nx
from PIL import Image

from description_to_image import generate_graph_image
from src.graph_title import generate_title, lookup_graph

ImageType = Image.Image


class Graph:
    """
    A graph abstraction that supports three interchangeable representations:
      - description: canonical representation (i.e., list of edges)
      - image: rendered diagram or visual object
      - title: human-readable name for the graph
    """

    def __init__(
        self,
        description: Optional[List[Tuple[int, int]]] = None,
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
    def _description_to_image(description: List[Tuple[int, int]]) -> ImageType:
        return generate_graph_image(description)

    @staticmethod
    def _description_to_title(description: List[Tuple[int, int]]) -> str:
        G = nx.Graph()
        G.add_edges_from(description)
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
    def _image_to_description(image: ImageType) -> Optional[List[Tuple[int, int]]]:
        """
        Convert a graph image to a list of edges (description).
        
        Uses the ImprovedGraphDetector to detect nodes and edges from the image.
        Returns a list of (source, target) tuples representing directed edges.
        
        Args:
            image: PIL Image containing a graph diagram
            
        Returns:
            List of (source, target) edge tuples, or None if detection fails
        """
        try:
            from image_to_description.improved_detector import ImprovedGraphDetector
            
            # Create detector directly with PIL Image
            detector = ImprovedGraphDetector(image)
            
            # Detect and get edges in one call
            return detector.detect_and_get_edges(min_radius=20, max_radius=100, detect_arrows=True)
            
        except Exception:
            # If detection fails, return None
            return None

    def __repr__(self) -> str:
        return (
            f"Graph(description={self.description}, " f"title={self.title}, " f"image={self.image})"
        )
