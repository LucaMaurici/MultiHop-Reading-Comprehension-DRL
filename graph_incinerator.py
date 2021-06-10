import pickle

graph_path = "CoreferenceGraphsList_dev3_uniqueAnswerFiltered.pkl"

SAMPLE_TO_DELETE = 3

try:
    with open(graph_path, 'rb') as f:
        graphs_list = pickle.load(f)
    f.close()
except:
    graphs_list = list()
    print("ERROR")

graphs_list.pop(3)


with open(graph_path, 'wb') as f:
    pickle.dump(graphs_list, f)
    print("\nSAVED\n")
f.close()

print("Item deleted")