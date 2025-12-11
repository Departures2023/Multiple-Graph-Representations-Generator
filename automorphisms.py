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
    """Convert various input types to NetworkX graph with weight support."""
    # Already a NetworkX graph
    if isinstance(graph_input, nx.Graph):
        return graph_input
    
    # List of edges (may be weighted or unweighted)
    if isinstance(graph_input, list):
        G = nx.Graph()
        for edge in graph_input:
            if len(edge) == 3:
                # Weighted edge: (u, v, weight)
                G.add_edge(edge[0], edge[1], weight=edge[2])
            elif len(edge) == 2:
                # Unweighted edge: (u, v)
                G.add_edge(edge[0], edge[1])
        return G
    
    # Your Graph class
    if hasattr(graph_input, 'description') and graph_input.description is not None:
        G = nx.Graph()
        for edge in graph_input.description:
            if len(edge) == 3:
                # Weighted edge
                G.add_edge(edge[0], edge[1], weight=edge[2])
            elif len(edge) == 2:
                # Unweighted edge
                G.add_edge(edge[0], edge[1])
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


# ==========================================
# GRAPH ISOMORPHISM FUNCTIONS
# ==========================================

def are_graphs_isomorphic(graph1_input, graph2_input) -> bool:
    """
    Check if two graphs are isomorphic.
    
    Args:
        graph1_input: First graph (NetworkX graph, edge list, or Graph object)
        graph2_input: Second graph (NetworkX graph, edge list, or Graph object)
    
    Returns:
        True if graphs are isomorphic, False otherwise
    """
    G1 = _to_networkx(graph1_input)
    G2 = _to_networkx(graph2_input)
    
    return nx.is_isomorphic(G1, G2)


def find_isomorphism(graph1_input, graph2_input) -> Dict:
    """
    Find an isomorphism mapping between two graphs if one exists.
    
    Args:
        graph1_input: First graph (NetworkX graph, edge list, or Graph object)
        graph2_input: Second graph (NetworkX graph, edge list, or Graph object)
    
    Returns:
        Dictionary mapping nodes from graph1 to graph2, or None if not isomorphic
    """
    G1 = _to_networkx(graph1_input)
    G2 = _to_networkx(graph2_input)
    
    matcher = nx.isomorphism.GraphMatcher(G1, G2)
    
    if matcher.is_isomorphic():
        return matcher.mapping
    else:
        return None


def get_all_isomorphisms(graph1_input, graph2_input) -> List[Dict]:
    """
    Get all possible isomorphism mappings between two graphs.
    
    Args:
        graph1_input: First graph (NetworkX graph, edge list, or Graph object)
        graph2_input: Second graph (NetworkX graph, edge list, or Graph object)
    
    Returns:
        List of dictionaries, each mapping nodes from graph1 to graph2
    """
    G1 = _to_networkx(graph1_input)
    G2 = _to_networkx(graph2_input)
    
    matcher = nx.isomorphism.GraphMatcher(G1, G2)
    
    return list(matcher.isomorphisms_iter())


def compare_graphs(graph1_input, graph2_input) -> Dict:
    """
    Compare two graphs and return detailed structural information.
    
    Args:
        graph1_input: First graph (NetworkX graph, edge list, or Graph object)
        graph2_input: Second graph (NetworkX graph, edge list, or Graph object)
    
    Returns:
        Dictionary with comparison results including:
        - are_isomorphic: bool (structural isomorphism)
        - same_size: bool
        - same_edges: bool
        - weights_match: bool (if isomorphic, do edge weights match?)
        - graph1_nodes: int
        - graph2_nodes: int
        - graph1_edges: int
        - graph2_edges: int
        - isomorphism_mapping: Dict or None
        - graph1_has_weights: bool
        - graph2_has_weights: bool
    """
    G1 = _to_networkx(graph1_input)
    G2 = _to_networkx(graph2_input)
    
    same_nodes = G1.number_of_nodes() == G2.number_of_nodes()
    same_edges = G1.number_of_edges() == G2.number_of_edges()
    
    # Check if graphs have weights
    g1_has_weights = any('weight' in data for _, _, data in G1.edges(data=True))
    g2_has_weights = any('weight' in data for _, _, data in G2.edges(data=True))
    
    is_isomorphic = False
    mapping = None
    weights_match = None
    
    if same_nodes and same_edges:
        # Check structural isomorphism (ignoring weights)
        is_isomorphic = nx.is_isomorphic(G1, G2)
        if is_isomorphic:
            mapping = find_isomorphism(graph1_input, graph2_input)
            
            # If both graphs have weights and are isomorphic, check if weights match
            if g1_has_weights and g2_has_weights and mapping:
                weights_match = True
                for u, v in G1.edges():
                    # Find corresponding edge in G2 using mapping
                    u2, v2 = mapping[u], mapping[v]
                    
                    # Get weights (handle both edge directions)
                    w1 = G1[u][v].get('weight', None)
                    w2 = G2.get_edge_data(u2, v2, default={}).get('weight', None)
                    if w2 is None:
                        # Try reverse direction
                        w2 = G2.get_edge_data(v2, u2, default={}).get('weight', None)
                    
                    # Compare weights (allowing small floating point differences)
                    if w1 is not None and w2 is not None:
                        if abs(w1 - w2) > 0.01:
                            weights_match = False
                            break
                    elif w1 != w2:  # One has weight, other doesn't
                        weights_match = False
                        break
    
    return {
        'are_isomorphic': is_isomorphic,
        'same_size': same_nodes and same_edges,
        'same_nodes': same_nodes,
        'same_edges': same_edges,
        'weights_match': weights_match,
        'graph1_nodes': G1.number_of_nodes(),
        'graph2_nodes': G2.number_of_nodes(),
        'graph1_edges': G1.number_of_edges(),
        'graph2_edges': G2.number_of_edges(),
        'graph1_has_weights': g1_has_weights,
        'graph2_has_weights': g2_has_weights,
        'isomorphism_mapping': mapping,
        'graph1_symmetries': count_automorphisms(G1),
        'graph2_symmetries': count_automorphisms(G2),
    }