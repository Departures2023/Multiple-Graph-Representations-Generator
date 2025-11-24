from PIL import Image

from graph import Graph

if __name__ == "__main__":
    # Graph 1: Test description -> image + title
    graph_1_description = [(1, 2), (2, 3), (3, 1)]
    graph_1_title = None
    graph_1_image = None

    g = Graph(description=graph_1_description, title=graph_1_title, image=graph_1_image)
    print(g)
    if g.image:
        g.image.show()

    # Graph 2: Test image -> description + title
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_image_path = os.path.join(script_dir, "image_to_description", "sample_complex.png")
    graph_2_image = Image.open(sample_image_path) if os.path.exists(sample_image_path) else None
    graph_2_description = Graph._image_to_description(graph_2_image) if graph_2_image else None
    graph_2_title = None

    g = Graph(description=graph_2_description, title=graph_2_title, image=graph_2_image)
    print(g)

    # Graph 3: Test title -> description + image
    graph_3_description = None
    graph_3_title = None
    graph_3_image = Image.new("RGB", (100, 100), color="white")  # temp

    g = Graph(description=graph_3_description, title=graph_3_title, image=graph_3_image)
    print(g)
