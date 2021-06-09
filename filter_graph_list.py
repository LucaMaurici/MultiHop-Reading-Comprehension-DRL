import pickle
import paths

TRAIN_MODE = False
UNIQUE_ANSWER_FILTERED = True

#--- Train ---
if TRAIN_MODE:
    graph_path = paths.graph_path_train
#--- Dev ---
else:
    graph_path = paths.graph_path_dev

try:
    with open(graph_path, 'rb') as f:
        graphs_list = pickle.load(f)
    f.close()
except:
    graphs_list = list()


graphs_list_length = len(graphs_list)
print(f"len(graphs_list) = {graphs_list_length}")

new_graph_list = list()

for i, elem in enumerate(graphs_list):
    #print(f"--- SAMPLE {i} ---")

    if UNIQUE_ANSWER_FILTERED:
        if len(elem["answer_positions"]) != 1:
            #--- Train ---
            if TRAIN_MODE:
                #file_name = "CoreferenceGraphsList_train_uniqueAnswerFiltered.pkl"
                file_name = paths.graph_path_train[0:-4] + "_uniqueAnswerFiltered.pkl"
            #--- Dev ---
            else:
                #file_name = "CoreferenceGraphsList_dev_uniqueAnswerFiltered.pkl"
                file_name = paths.graph_path_dev[0:-4] + "_uniqueAnswerFiltered.pkl"
            continue
    else:
        if len(elem[answer_positions]) == 0:
            #--- Train ---
            if TRAIN_MODE:
                #file_name = "CoreferenceGraphsList_train_multipleAnswersFiltered.pkl"
                file_name = paths.graph_path_train[0:-4] + "_multipleAnswersFiltered.pkl"
            #--- Dev ---
            else:
                #file_name = "CoreferenceGraphsList_dev_multipleAnswersFiltered.pkl"
                file_name = paths.graph_path_dev[0:-4] + "_multipleAnswersFiltered.pkl"
            continue

    new_graph_list.append(elem)
    #print("ACCEPTED")


with open(file_name, 'wb') as f:
    pickle.dump(new_graph_list, f)
    print("\nSAVED\n")
f.close()

with open(file_name, 'rb') as f:
    new_graph_list = list(pickle.load(f))
    #print(f"\n---GRAPH_LIST---\n {new_graph_list}")
    new_graph_list_length = len(new_graph_list)
    print(f"len(graphs_list) = {new_graph_list_length}")
f.close()


