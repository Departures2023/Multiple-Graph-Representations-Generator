"""
graph_title.py
---------------
Improved graph naming and title generation for small graphs (≤ 20 nodes).
"""

import networkx as nx
from typing import Dict, Optional

def build_title_db() -> Dict[str, str]:
    """
    Build a comprehensive title database from known named graphs.
    Uses a curated list of important graphs rather than auto-discovery.
    """    
    title_db = {}

    # Helper to add graph to database
    def add_graph(title: str, G: nx.Graph):
        if G.number_of_nodes() <= 20:
            g6 = nx.to_graph6_bytes(G.to_undirected(), header=False).decode().strip()
            title_db[g6] = title

    # Cycle graphs
    for n in range(3, 21):
        add_graph(f"Cycle graph C{n}", nx.cycle_graph(n))
    
    # Path graphs
    for n in range(2, 21):
        add_graph(f"Path graph P{n}", nx.path_graph(n))
    
    # Complete graphs
    for n in range(2, 21):
        add_graph(f"Complete graph K{n}", nx.complete_graph(n))
    
    # Complete bipartite graphs (common ones)
    for m in range(1, 8):
        for n in range(m, 8):
            if m + n <= 20:
                add_graph(f"Complete bipartite graph K{m},{n}", 
                         nx.complete_bipartite_graph(m, n))
    
    # Star graphs
    for n in range(3, 21):
        add_graph(f"Star graph S{n}", nx.star_graph(n))
    
    # Wheel graphs
    for n in range(4, 21):
        add_graph(f"Wheel graph W{n}", nx.wheel_graph(n))
    
    # Regular graphs and special named graphs
    special_graphs = {
        "Petersen graph": nx.petersen_graph(),
        "Tutte graph": nx.tutte_graph(),
        "Dodecahedral graph": nx.dodecahedral_graph(),
        "Icosahedral graph": nx.icosahedral_graph(),
        "Octahedral graph": nx.octahedral_graph(),
        "Tetrahedral graph": nx.tetrahedral_graph(),
        "Cubical graph": nx.cubical_graph(),
        "Desargues graph": nx.desargues_graph(),
        "Heawood graph": nx.heawood_graph(),
        "Pappus graph": nx.pappus_graph(),
        "Frucht graph": nx.frucht_graph(),
        "House graph": nx.house_graph(),
        "House X graph": nx.house_x_graph(),
        "Bull graph": nx.bull_graph(),
        "Chvatal graph": nx.chvatal_graph(),
        "Diamond graph": nx.diamond_graph(),
        "Krackhardt kite graph": nx.krackhardt_kite_graph(),
        "Moebius-Kantor graph": nx.moebius_kantor_graph(),
        "Truncated cube graph": nx.truncated_cube_graph(),
        "Truncated tetrahedron graph": nx.truncated_tetrahedron_graph(),
    }
    
    for title, G in special_graphs.items():
        add_graph(title, G)
    
    # Grid graphs
    for m in range(2, 11):
        for n in range(m, 11):
            if m * n <= 20:
                add_graph(f"Grid graph {m}×{n}", nx.grid_2d_graph(m, n))
    
    # Hypercube graphs
    for n in range(1, 5):  # Q1 through Q4 (2, 4, 8, 16 nodes)
        add_graph(f"Hypercube graph Q{n}", nx.hypercube_graph(n))
    
    # Paley graphs (primes only)
    for p in [5, 9, 13, 17]:
        try:
            add_graph(f"Paley graph ({p})", nx.paley_graph(p))
        except:
            pass
    
    # Ladder graphs
    for n in range(2, 11):
        if 2 * n <= 20:
            add_graph(f"Ladder graph L{n}", nx.ladder_graph(n))
    
    # Circular ladder graphs
    for n in range(3, 11):
        if 2 * n <= 20:
            add_graph(f"Circular ladder graph CL{n}", nx.circular_ladder_graph(n))
    
    # Barbell graphs
    for m in range(2, 8):
        for n in range(0, 5):
            if 2 * m + n <= 20:
                add_graph(f"Barbell graph B{m},{n}", nx.barbell_graph(m, n))

    return title_db


# Build database once at module load
TITLE_DB = build_title_db()


def generate_title(G: nx.Graph, check_isomorphism: bool = True) -> str:
    """
    Generate a title for a NetworkX graph.
    
    Args:
        G: Input graph
        check_isomorphism: If True, perform slower isomorphism check if no direct match
    
    Returns:
        Graph title (either canonical name or synthetic description)
    """
    # Ensure graph size constraint
    if G.number_of_nodes() > 20:
        raise ValueError("Graph exceeds 20-node limit.")
    
    if G.number_of_nodes() == 0:
        return "Empty graph"
    
    if G.number_of_nodes() == 1:
        return "Trivial graph (1 node)"
    
    # Convert to undirected and compute canonical encoding
    G_undirected = G.to_undirected() if G.is_directed() else G
    g6 = nx.to_graph6_bytes(G_undirected, header=False).decode().strip()
    
    # Check for direct match
    if g6 in TITLE_DB:
        return TITLE_DB[g6]
    
    # Optional: Check for isomorphic matches (slower)
    if check_isomorphism and G.number_of_nodes() <= 10:
        for known_g6, title in TITLE_DB.items():
            try:
                known_graph = nx.from_graph6_bytes((known_g6 + '\n').encode())
                if nx.is_isomorphic(G_undirected, known_graph):
                    return title
            except:
                continue
    
    # Generate synthetic descriptor
    return _generate_synthetic_title(G_undirected)


def _generate_synthetic_title(G: nx.Graph) -> str:
    """Generate a concise synthetic title for an unknown graph."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    # Check for special structures
    if m == 0:
        return f"Empty graph ({n} nodes)"
    
    if m == n * (n - 1) // 2:
        return f"Complete graph K{n}"
    
    degrees = sorted([d for _, d in G.degree()])
    min_deg, max_deg = degrees[0], degrees[-1]
    
    # Check regularity
    if min_deg == max_deg:
        if min_deg == 2 and nx.is_connected(G):
            return f"Cycle graph C{n}"
        return f"{min_deg}-regular graph ({n} nodes, {m} edges)"
    
    # Check bipartiteness
    if nx.is_bipartite(G):
        return f"Bipartite graph ({n} nodes, {m} edges)"
    
    # Check tree
    if nx.is_tree(G):
        return f"Tree ({n} nodes)"
    
    # Check connectivity
    connected = "connected" if nx.is_connected(G) else "disconnected"
    
    # Default descriptor
    return f"{connected.capitalize()} graph ({n} nodes, {m} edges, deg {min_deg}-{max_deg})"


def lookup_graph(name: str) -> Optional[nx.Graph]:
    """
    Reverse lookup: given a graph name, return the corresponding graph.
    
    Args:
        name: Graph name (case-insensitive, partial matching supported)
    
    Returns:
        NetworkX graph or None if not found
    """
    name_lower = name.lower()
    
    # Try exact match first
    for g6, title in TITLE_DB.items():
        if title.lower() == name_lower:
            return nx.from_graph6_bytes((g6 + '\n').encode())
    
    # Try partial match
    matches = []
    for g6, title in TITLE_DB.items():
        if name_lower in title.lower():
            matches.append((title, g6))
    
    if len(matches) == 1:
        return nx.from_graph6_bytes((matches[0][1] + '\n').encode())
    elif len(matches) > 1:
        print(f"Multiple matches found: {[m[0] for m in matches]}")
        return None
    
    return None


# Example usage and testing
if __name__ == "__main__":
    # Test with known graphs
    print("Testing known graphs:")
    print(generate_title(nx.petersen_graph()))
    print(generate_title(nx.cycle_graph(5)))
    print(generate_title(nx.complete_graph(4)))
    
    # Test with custom graph
    print("\nTesting custom graph:")
    G = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0)])
    print(generate_title(G))
    
    # Test reverse lookup
    print("\nTesting reverse lookup:")
    G = lookup_graph("petersen")
    if G:
        print(f"Found: {generate_title(G)}, {G.number_of_nodes()} nodes")