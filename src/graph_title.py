"""
graph_title.py
---------------
Handles canonical naming and synthetic title generation for small graphs (≤ 20 nodes).
Integrates with NetworkX; can export graphs to LaTeX (TikZ) or image formats.
"""
import inspect
import networkx as nx

# Optional: later you can add 'pygraphviz' or 'network2tikz' for visualization/export
# import pygraphviz as pgv
# from network2tikz import plot as tikz_plot


# ===============================
# 1. GRAPH TITLE DATABASE
# ===============================

# auto_discover asks whether it should explore other potential graphs
def build_title_db(auto_discover=True):
    """
    Build a canonical title database from known named graphs in NetworkX.
    - Automatically loads graphs from common NetworkX generator modules.
    - Skips functions that require parameters.
    """    
    title_db = {}

    # Base known graphs you want to guarantee
    known_graphs = {
        "Cycle graph C4": nx.cycle_graph(4),
        "Cycle graph C5": nx.cycle_graph(5),
        "Complete graph K5": nx.complete_graph(5),
        "Complete bipartite graph K3,3": nx.complete_bipartite_graph(3, 3),
        "Paley graph (9)": nx.paley_graph(9),
    }

    # Always include these
    for title, G in known_graphs.items():
        g6 = nx.to_graph6_bytes(G.to_undirected()).decode().strip()
        title_db[g6] = title

    # Auto-discover other named graphs if requested
    if auto_discover:
        generator_modules = [
            nx.generators.classic,
            nx.generators.small,
            nx.generators.social,
            nx.generators.atlas
        ]

        for module in generator_modules:
            for name, func in inspect.getmembers(module, inspect.isfunction):
                # Only call functions that take 0 arguments (safe)
                try:
                    sig = inspect.signature(func)
                    if all(p.default != inspect.Parameter.empty or p.kind == inspect.Parameter.VAR_KEYWORD
                           for p in sig.parameters.values()):
                        # Function has all default args — safe to call
                        G = func()
                        if isinstance(G, nx.Graph) and G.number_of_nodes() > 0:
                            title = name.replace("_", " ").capitalize()
                            g6 = nx.to_graph6_bytes(G).decode().strip()
                            title_db[g6] = title
                except Exception:
                    # Ignore anything that fails
                    continue

    return title_db



TITLE_DB = build_title_db()


# ===============================
# 2. TITLE GENERATOR
# ===============================

def generate_title(G: nx.Graph) -> str:
    """
    Generate a canonical or synthetic title for a given NetworkX graph object.
    If a known isomorphic graph is found, return its canonical title.
    Otherwise, return a descriptive synthetic title.
    """
    # Ensure graph size constraint
    if G.number_of_nodes() > 20:
        raise ValueError("Graph exceeds 20-node limit.")

    # Compute canonical encoding
    g6 = nx.to_graph6_bytes(G).decode().strip()

    # Check canonical match
    if g6 in TITLE_DB:
        return TITLE_DB[g6]

    # Check for isomorphic matches (in case label differences)
    for known_g6, title in TITLE_DB.items():
        known_graph = nx.from_graph6_bytes(known_g6.encode())
        if nx.is_isomorphic(G, known_graph):
            return title

    # Fallback synthetic descriptor
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    regular = all(d == degrees[0] for d in degrees)
    planar = nx.check_planarity(G)[0]
    bipartite = nx.is_bipartite(G)

    desc = []
    desc.append("Planar" if planar else "Nonplanar")
    if regular:
        desc.append(f"{degrees[0]}-regular")
    if bipartite:
        desc.append("bipartite")
    desc.append(f"{n}-node, {m}-edge graph")

    return " ".join(desc)


# ===============================
# 3. (OPTIONAL) EXPORTERS
# ===============================

def export_graph_latex(G: nx.Graph, filepath: str = "graph.tex"):
    """
    Export a NetworkX graph to LaTeX/TikZ code using network2tikz (optional).
    """
    try:
        from network2tikz import plot
        plot(G, filename=filepath)
        print(f"Graph exported to {filepath}")
    except ImportError:
        print("network2tikz not installed. Run `pip install network2tikz` to enable LaTeX export.")


def export_graph_image(G: nx.Graph, filepath: str = "graph.png", layout_prog: str = "dot"):
    """
    Export a NetworkX graph to an image using Graphviz layout.
    """
    try:
        import pygraphviz as pgv
        A = nx.nx_agraph.to_agraph(G)
        A.draw(filepath, prog=layout_prog)
        print(f"Graph image saved to {filepath}")
    except ImportError:
        print("pygraphviz not installed. Run `pip install pygraphviz` to enable image export.")
