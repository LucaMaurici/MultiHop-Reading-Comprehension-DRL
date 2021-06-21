import pickle
import paths

TRAIN_MODE = False

#--- Train ---
if TRAIN_MODE:
    graph_path_output = paths.graph_path_train
#--- Dev ---
else:
    graph_path_output = paths.graph_path_dev

graph_path_0 = "CoreferenceGraphsList_dev3_0.pkl"

try:
    with open(graph_path_0, 'rb') as f:
        graphs_list_0 = pickle.load(f)
    f.close()
except:
    graphs_list_0 = list()


graph_path_1 = "CoreferenceGraphsList_dev3_1.pkl"

try:
    with open(graph_path_1, 'rb') as f:
        graphs_list_1 = pickle.load(f)
    f.close()
except:
    graphs_list_1 = list()


def contains_graph(graphs_list, elem_tocheck):
    for elem in graphs_list:
        if elem["id"] == elem_tocheck["id"]:
            return True

    return False

def remove_duplicate(graphs_list):
    graph_list_output = list()
    for elem in graphs_list:
        if not contains_graph(graph_list_output, elem):
            graph_list_output.append(elem)

    return graph_list_output


graphs_list_tot = graphs_list_0 + graphs_list_1;
print(f"\nGraphs_list_0: {len(graphs_list_0)}\n")
print(f"\nGraphs_list_1: {len(graphs_list_1)}\n")
print(f"\nGraphs_list_tot: {len(graphs_list_tot)}\n")

graphs_list_noduplicates = remove_duplicate(graphs_list_tot)

print(f"\nGraphs_list_noduplicates: {len(graphs_list_noduplicates)}\n")

with open(graph_path_output, 'wb') as f:
    pickle.dump(graphs_list_noduplicates, f)
f.close()