import pickle
import networkx as nx
import matplotlib.pyplot as plt

import MultiHopEnvironment
import paths
#import random
import math
import random


def value(x):
    #return round( (12/11) - ( math.exp(x) / (10 + math.exp(x)) ), 3 )
    #return round( (7/6) - ( math.exp(x) / (5 + math.exp(x)) ), 3 )
    return round(1-0.25*x, 2)

def graph_plot(G, node2value):
    values = [node2value[node] for node in G.nodes()]
    values = [int(value*5) for value in values]
    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Set2'), 
                           node_color = values, node_size = 500)
    nx.draw_networkx_labels(G, pos, node2value)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='g', arrows=True)

    plt.show()

graph_path = "CoreferenceGraphsList_train3_uniqueAnswerFiltered.pkl"
graph_marked_path = "CoreferenceGraphsList_train3_uniqueAnswerFiltered_marked.pkl"

skip = [68, 418, 550, 605, 832, 852, 915, 990, 1124, 1183, 1241, 1545, 1812, 1944]

try:
    with open(graph_path, 'rb') as f:
        graphs_list = pickle.load(f)
    f.close()
except:
    graphs_list = list()

graphs_marked_list = list()

for i, elem in enumerate(graphs_list):

    if i in skip:
        continue

    graph = elem["graph"]

    G = nx.DiGraph()

    G.add_edges_from(graph.getEdges())

    shortest_path_length = nx.shortest_path_length(G, source='q', target=elem["answer_positions"][0], weight=None, method='dijkstra') #dijkstra
    print(f"{i}. shortest_path_length: {shortest_path_length}")
    #shortest_paths = nx.shortest_paths(G, 'q', elem["answer_positions"][0], weight=None, method='dijkstra')
    node2value = {}
    for node in graph.getNodes():
        node2value[node] = 0
    for length in range(shortest_path_length+3, shortest_path_length-1, -1):
        #if shortest_path_length*2 - shortest_path_length > 4:
            #shortest_paths = nx.shortest_paths(G, 'q', elem["answer_positions"][0], weight=None, method='dijkstra')
            #print(shortest_path_length)
            #length = 8
        
        all_simple_paths = nx.all_simple_paths(G, 'q', elem["answer_positions"][0], cutoff=length)

        for path in all_simple_paths:
            for node in path:
                node2value[node] = value(length-shortest_path_length)

    randomNumber = random.randint(0, 300)
    if randomNumber == 0:
        graph_plot(G, node2value)

    elem["node2value"] = node2value

    graphs_marked_list.append(elem)

    #print(f"\nall_shortest_paths: {[p for p in all_shortest_paths]}")
    #print(f"\nall_simple_paths: {[p for p in all_simple_paths]}")


with open(graph_marked_path, 'wb') as f:
    pickle.dump(graphs_marked_list, f)
f.close()

