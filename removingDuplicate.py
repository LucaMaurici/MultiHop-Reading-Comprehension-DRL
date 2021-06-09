import pickle
import paths
import paths


TRAIN_MODE = False

#--- Train ---
if TRAIN_MODE:
    graph_path = paths.graph_path_train
    dataset_path = paths.dataset_path_train
#--- Dev ---
else:
    graph_path = paths.graph_path_dev
    dataset_path = paths.dataset_path_dev

'''
with open(dataset_path, "r") as read_file:
    dataset = json.load(read_file)
read_file.close()

try:
    with open(graph_path, 'rb') as f:
        graphs_list = pickle.load(f)
    f.close()
except:
    graphs_list = list()
'''

graph_path = "CoreferenceGraphsList_dev_uniqueAnswerFiltered.pkl"

try:
    with open(graph_path, 'rb') as f:
        graphs_list = pickle.load(f)
    f.close()
except:
    graphs_list = list()



graph_path1 = "CoreferenceGraphsList_dev_uniqueAnswerFiltered1.pkl"

try:
    with open(graph_path, 'rb') as f:
        graphs_list1 = pickle.load(f)
    f.close()
except:
    graphs_list1 = list()



def removeDuplicate(graphs_list):
    graphs_set = set()
    new_graphs_list = []
    for d in graphs_list:
        t = tuple(d.items())
        if t not in graphs_set:
            graphs_list.add(t)
            new__graphs_list.append(d)

    return new__graphs_list



graphs_list_noduplicates = removeDuplicate(graphs_list)
graphs_list_noduplicates1 = removeDuplicate(graphs_list1)

graphs_list_noduplicate_tot = graphs_list_noduplicates +  graphs_list_noduplicates1;

with open(graph_path, 'wb') as f:
        pickle.dump(graphs_list, f)
    f.close()