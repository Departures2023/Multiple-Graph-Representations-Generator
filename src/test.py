## use this file to test any changes to src

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_title import generate_title, TITLE_DB
import networkx as nx

print(len(TITLE_DB)) 

print(generate_title(nx.petersen_graph()))
print(generate_title(nx.heawood_graph()))
print(generate_title(nx.random_regular_graph(3, 10)))
