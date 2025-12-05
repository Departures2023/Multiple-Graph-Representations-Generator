"""
automorphism.py
---------------
Counts automorphisms (symmetries) of graphs.

An automorphism is a permutation of nodes that preserves edge structure.
For example, a square has 8 automorphisms (4 rotations + 4 reflections).
"""

from typing import List, Dict
import networkx as nx


def count_automorphisms(graph_input) -> int:
    """
    Count automorphisms using NetworkX's VF2 algorithm.
    
    Args:
        graph_input: NetworkX graph, edge list, or Graph object
    
    Returns:
        Number of automorphisms (always ≥ 1)
    """
    G = _to_networkx(graph_input)
    
    if G.number_of_nodes() == 0:
        return 1
    
    matcher = nx.isomorphism.GraphMatcher(G, G)
    return sum(1 for _ in matcher.isomorphisms_iter())


def get_all_automorphisms(graph_input) -> List[Dict]:
    """
    Get all automorphisms as a list of node mappings.
    
    Args:
        graph_input: NetworkX graph, edge list, or Graph object
    
    Returns:
        List of dictionaries, where each dict maps original_node -> new_node
    """
    G = _to_networkx(graph_input)
    
    if G.number_of_nodes() == 0:
        return [{}]
    
    matcher = nx.isomorphism.GraphMatcher(G, G)
    return list(matcher.isomorphisms_iter())


def describe_symmetry(graph_input) -> Dict:
    """
    Get detailed information about the graph's symmetry.
    
    Args:
        graph_input: NetworkX graph, edge list, or Graph object
    
    Returns:
        Dictionary with symmetry information
    """
    G = _to_networkx(graph_input)
    auto_count = count_automorphisms(G)
    vertex_transitive = _is_vertex_transitive(G)
    
    return {
        'automorphism_count': auto_count,
        'is_vertex_transitive': vertex_transitive,
        'is_asymmetric': auto_count == 1,
    }


def _to_networkx(graph_input):
    """Convert various input types to NetworkX graph."""
    # Already a NetworkX graph
    if isinstance(graph_input, nx.Graph):
        return graph_input
    
    # List of edges
    if isinstance(graph_input, list):
        G = nx.Graph()
        G.add_edges_from(graph_input)
        return G
    
    # Your Graph class
    if hasattr(graph_input, 'description') and graph_input.description is not None:
        G = nx.Graph()
        G.add_edges_from(graph_input.description)
        return G
    
    raise TypeError(f"Cannot convert {type(graph_input)} to NetworkX graph")


def _is_vertex_transitive(G: nx.Graph) -> bool:
    """Check if graph is vertex-transitive."""
    if G.number_of_nodes() == 0:
        return True
    
    orbits = _count_orbits(G)
    return orbits == 1


def _count_orbits(G: nx.Graph) -> int:
    """Count the number of orbits of vertices under the automorphism group."""
    if G.number_of_nodes() == 0:
        return 0
    
    nodes = list(G.nodes())
    visited = set()
    orbit_count = 0
    
    automorphisms = get_all_automorphisms(G)
    
    for node in nodes:
        if node in visited:
            continue
        
        # Find orbit of this node
        orbit = {node}
        for auto in automorphisms:
            orbit.add(auto[node])
        
        visited.update(orbit)
        orbit_count += 1
    
    return orbit_count
