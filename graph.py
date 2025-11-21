from __future__ import annotations

from typing import List, Optional, Tuple

from PIL import Image

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

        raise NotImplementedError

    # -----------------------------------------------------------------------
    # CONVERSION METHODS
    # These should be implemented.
    # -----------------------------------------------------------------------

    @staticmethod
    def _description_to_image(description: List[Tuple[int, int]]) -> ImageType:
        raise NotImplementedError

    @staticmethod
    def _description_to_title(description: List[Tuple[int, int]]) -> str:
        raise NotImplementedError

    @staticmethod
    def _title_to_description(title: str) -> Optional[List[Tuple[int, int]]]:
        raise NotImplementedError

    @staticmethod
    def _image_to_description(image: ImageType) -> Optional[List[Tuple[int, int]]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"Graph(description={self.description}, " f"title={self.title}, " f"image={self.image})"
        )
