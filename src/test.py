"""
test_graph_title.py
-------------------
Testing suite for the graph title generator.
Run this file to test various graphs and edge cases.
"""

import networkx as nx
from graph_title import generate_title, lookup_graph, TITLE_DB

def test_known_graphs():
    """Test graphs that should have canonical names."""
    print("=" * 60)
    print("TESTING KNOWN NAMED GRAPHS")
    print("=" * 60)
    
    test_cases = [
        (nx.petersen_graph(), "Petersen graph"),
        (nx.cycle_graph(5), "Cycle graph C5"),
        (nx.complete_graph(4), "Complete graph K4"),
        (nx.path_graph(5), "Path graph P5"),
        (nx.wheel_graph(6), "Wheel graph W6"),
        (nx.star_graph(5), "Star graph S5"),
        (nx.complete_bipartite_graph(3, 3), "Complete bipartite graph K3,3"),
        (nx.dodecahedral_graph(), "Dodecahedral graph"),
        (nx.house_graph(), "House graph"),
        (nx.diamond_graph(), "Diamond graph"),
        (nx.cubical_graph(), "Cubical graph"),
        (nx.tetrahedral_graph(), "Tetrahedral graph"),
    ]
    
    passed = 0
    for G, expected in test_cases:
        result = generate_title(G)
        status = "✓" if result == expected else "✗"
        print(f"{status} {expected:40s} -> {result}")
        if result == expected:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(test_cases)}")


def test_custom_graphs():
    """Test custom graphs that should get synthetic titles."""
    print("\n" + "=" * 60)
    print("TESTING CUSTOM GRAPHS (synthetic titles)")
    print("=" * 60)
    
    # Create various custom graphs
    test_graphs = [
        ("Square (cycle of 4)", nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0)])),
        ("Triangle", nx.Graph([(0, 1), (1, 2), (2, 0)])),
        ("Simple tree", nx.Graph([(0, 1), (0, 2), (1, 3), (1, 4)])),
        ("Disconnected (2 triangles)", nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))),
        ("5-node star", nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])),
        ("Empty graph (5 nodes)", nx.Graph([(i, i) for i in range(5)])),  # Self-loops ignored
        ("Path of 4", nx.Graph([(0, 1), (1, 2), (2, 3)])),
        ("3-regular graph", nx.cubical_graph()),  # Should recognize as canonical
    ]
    
    for description, G in test_graphs:
        # Remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        result = generate_title(G)
        print(f"{description:30s} -> {result}")


def test_isomorphism():
    """Test that isomorphic graphs with different labels are detected."""
    print("\n" + "=" * 60)
    print("TESTING ISOMORPHISM DETECTION")
    print("=" * 60)
    
    # Create a cycle with labels 0,1,2,3,4
    cycle1 = nx.cycle_graph(5)
    print(f"Cycle with labels 0-4: {generate_title(cycle1)}")
    
    # Create same cycle with labels 10,11,12,13,14
    cycle2 = nx.Graph([
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 10)
    ])
    print(f"Cycle with labels 10-14: {generate_title(cycle2, check_isomorphism=True)}")
    print("(Note: Isomorphism check is slow, only done for graphs ≤10 nodes)")
    
    # Without isomorphism check
    result_no_iso = generate_title(cycle2, check_isomorphism=False)
    print(f"Same cycle without isomorphism check: {result_no_iso}")


def test_edge_cases():
    """Test edge cases and special graphs."""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)
    
    # Single node
    G1 = nx.Graph()
    G1.add_node(0)
    print(f"Single node: {generate_title(G1)}")
    
    # Empty graph
    G2 = nx.Graph()
    print(f"Empty graph: {generate_title(G2)}")
    
    # Disconnected components
    G3 = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    print(f"Two disconnected triangles: {generate_title(G3)}")
    
    # Tree
    G4 = nx.balanced_tree(2, 3)  # Binary tree of height 3
    print(f"Binary tree: {generate_title(G4)}")
    
    # Complete graph
    G5 = nx.complete_graph(7)
    print(f"Complete graph K7: {generate_title(G5)}")


def test_reverse_lookup():
    """Test looking up graphs by name."""
    print("\n" + "=" * 60)
    print("TESTING REVERSE LOOKUP")
    print("=" * 60)
    
    queries = [
        "petersen",
        "Cycle graph C5",
        "complete graph k4",
        "house",
        "nonexistent graph",
    ]
    
    for query in queries:
        G = lookup_graph(query)
        if G:
            print(f"'{query}' -> Found: {generate_title(G)} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        else:
            print(f"'{query}' -> Not found")


def show_database_stats():
    """Show statistics about the graph database."""
    print("\n" + "=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)
    
    print(f"Total graphs in database: {len(TITLE_DB)}")
    
    # Count by type
    types = {}
    for title in TITLE_DB.values():
        graph_type = title.split()[0] + " " + title.split()[1] if len(title.split()) > 1 else title.split()[0]
        types[graph_type] = types.get(graph_type, 0) + 1
    
    print("\nGraphs by type:")
    for gtype, count in sorted(types.items(), key=lambda x: -x[1])[:10]:
        print(f"  {gtype:30s}: {count}")

if __name__ == "__main__":
    # Run all tests
    test_known_graphs()
    test_custom_graphs()
    test_isomorphism()
    test_edge_cases()
    test_reverse_lookup()
    show_database_stats()
    demonstrate_graph6()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE!")
    print("=" * 60)