import pickle
import networkx as nx
import matplotlib.pyplot as plt

import MultiHopEnvironment
import paths
import json
#import random

graph_path = "CoreferenceGraphsList_dev2_uniqueAnswerFiltered.pkl"

try:
    with open(graph_path, 'rb') as f:
        graphs_list = pickle.load(f)
    f.close()
except:
    graphs_list = list()

dataset_path = paths.dataset_path_dev

with open(dataset_path, "r") as read_file:
    dataset = json.load(read_file)
read_file.close()

#elem = graphs_list[1422]
elem = graphs_list[0]

graph = elem["graph"]

print(graph.getEdges())


print(MultiHopEnvironment.getSampleById(dataset, elem["id"])["query"])
print(elem["id2sentence"][elem["answer_positions"][0]])


G = nx.DiGraph()

G.add_edges_from(graph.getEdges())

val_dict = {'q': 1.0}
           #'D': 0.5714285714285714,
           #'H': 0.0}

val_dict[elem["answer_positions"][0]] = 0

values = [val_dict.get(node, 0.25) for node in G.nodes()]
'''
values = list()
random_number = random.randrange(2, 100)
doc_num = 0
nodes = list()
for node in G.nodes():
    nodes.append(node)
nodes.sort()
val_dict_others = {}
for node in nodes:
    if node != 'q' and node[1] != doc_num:
        doc_num = node[1]
        random_number = random.randrange(2, 100)
    val_dict_others[node] = random_number

values = [val_dict.get(node, val_dict_others[node]) for node in G.nodes()]
#values.append(val_dict.get(node, random_number))
'''

pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Set2'), 
                       node_color = values, node_size = 500)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, edgelist=graph.getEdges(), edge_color='g', arrows=True)

plt.show()

