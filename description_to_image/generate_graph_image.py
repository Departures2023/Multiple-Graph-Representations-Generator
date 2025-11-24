import networkx as nx
import matplotlib.pyplot as plt

from typing import List, Tuple
from io import BytesIO
from PIL import Image

# Construct a NetworkX graph and return a PIL Image object, given a list of edges [(u, v), ...]
def generate_graph_image(edges: List[Tuple[int, int]]) -> Image.Image:
    # 1. Build graph
    G = nx.Graph()
    G.add_edges_from(edges)

    # 2. Choose layout
    # use circular for simple cycles, spring otherwise
    if _is_cycle(G):
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # 3. Draw graph
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_axis_off()
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=800,
        font_size=10,
        width=1.5,
        ax=ax,
    )

    # 4. Export to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)

# Detect simple cycle graph
def _is_cycle(G: nx.Graph) -> bool:
    
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0 or m != n:
        return False

    for _, deg in G.degree():
        if deg != 2:
            return False

    return True
