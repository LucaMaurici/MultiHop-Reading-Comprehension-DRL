import pickle
import paths
import paths


TRAIN_MODE = False

#--- Train ---
if TRAIN_MODE:
    graph_path_output = paths.graph_path_train
#--- Dev ---
else:
    graph_path_output = paths.graph_path_dev

graph_path_0 = "CoreferenceGraphsList_dev2_0.pkl"

try:
    with open(graph_path_0, 'rb') as f:
        graphs_list_0 = pickle.load(f)
    f.close()
except:
    graphs_list_0 = list()


graph_path_1 = "CoreferenceGraphsList_dev2_1.pkl"

try:
    with open(graph_path_1, 'rb') as f:
        graphs_list_1 = pickle.load(f)
    f.close()
except:
    graphs_list_1 = list()


def removeDuplicate(graphs_list):
    graphs_set = set()
    new_graphs_list = list()
    for d in graphs_list:
        t = tuple(d.items())
        if t not in graphs_set:
            graphs_set.add(t)
            new_graphs_list.append(d)

    return new_graphs_list


graphs_list_tot = graphs_list_0 +  graphs_list_1;
graphs_list_noduplicates = removeDuplicate(graphs_list_tot)

with open(graph_path_output, 'wb') as f:
    pickle.dump(graphs_list_noduplicates, f)
f.close()