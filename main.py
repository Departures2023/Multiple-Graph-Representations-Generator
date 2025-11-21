from PIL import Image

from graph import Graph

if __name__ == "__main__":
    # Graph 1: Test description -> image + title
    graph_1_description = [(1, 2), (2, 3), (3, 1)]
    graph_1_title = None
    graph_1_image = None

    g = Graph(description=graph_1_description, title=graph_1_title, image=graph_1_image)
    print(g)

    # Graph 2: Test image -> description + title
    graph_2_description = None
    graph_2_title = "C4 Graph"
    graph_2_image = None

    g = Graph(description=graph_2_description, title=graph_2_title, image=graph_2_image)
    print(g)

    # Graph 3: Test title -> description + image
    graph_3_description = None
    graph_3_title = None
    graph_3_image = Image.new("RGB", (100, 100), color="white")  # temp

    g = Graph(description=graph_3_description, title=graph_3_title, image=graph_3_image)
    print(g)
