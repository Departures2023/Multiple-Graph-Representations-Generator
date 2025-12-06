import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple
from io import BytesIO
from PIL import Image

import matplotlib
matplotlib.use("Agg")


# Graph Type Detectors
def is_cycle(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    return n >= 3 and m == n and all(deg == 2 for _, deg in G.degree())

def is_path(G):
    degrees = [deg for _, deg in G.degree()]
    return degrees.count(1) == 2 and degrees.count(2) == len(degrees) - 2

def is_star(G):
    n = G.number_of_nodes()
    if n <= 2:
        return False
    center = [node for node, deg in G.degree() if deg == n - 1]
    return len(center) == 1

def is_tree(G):
    return nx.is_tree(G)

def is_bipartite_graph(G):
    return nx.is_bipartite(G)

def is_planar(G):
    try:
        planar, _ = nx.check_planarity(G)
        return planar
    except:
        return False

# Layout Helpers 
def path_layout(G):
    nodes = list(G.nodes())
    return {v: (i, 0) for i, v in enumerate(nodes)}


def hierarchy_layout(G, root=None):
    if root is None:
        root = list(G.nodes())[0]

    pos = {}

    def dfs(node, x, y, visited, spread):
        visited.add(node)
        pos[node] = (x, y)
        children = [n for n in G.neighbors(node) if n not in visited]

        if not children:
            return

        step = spread / max(1, len(children))
        start = x - spread / 2 + step / 2

        for i, child in enumerate(children):
            dfs(child, start + step * i, y - 1, visited, spread / 1.5)

    dfs(root, 0, 0, set(), 3.0)
    return pos

# Choose Layout
def choose_layout(G):

    if is_cycle(G):
        return nx.circular_layout(G)

    if is_path(G):
        return path_layout(G)

    if is_star(G):
        return nx.spring_layout(G, seed=42, k=1.8)

    if is_tree(G):
        return hierarchy_layout(G)

    if is_bipartite_graph(G):
        left, _ = nx.bipartite.sets(G)
        return nx.bipartite_layout(G, left)

    if is_planar(G) and G.number_of_nodes() <= 20:
        return nx.planar_layout(G)

    return nx.spring_layout(G, seed=42, k=1.2)

# Main Render Function
def generate_graph_image(edges: List[Tuple[int, int]]) -> Image.Image:
    """
    Build a graph from edge list and return a PIL image.
    """
    G = nx.Graph()
    G.add_edges_from(edges)

    pos = choose_layout(G)

    # Clean visual defaults
    NODE_SIZE = 1100
    FONT_SIZE = 12
    EDGE_WIDTH = 2.0

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_axis_off()

    nx.draw(
        G,
        pos=pos,
        ax=ax,
        with_labels=True,
        node_color="#8ecae6",
        node_size=NODE_SIZE,
        width=EDGE_WIDTH,
        font_size=FONT_SIZE,
        edge_color="black",
    )

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)
