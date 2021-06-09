import pickle
import CoreferenceGraph as cg
import json
import random
import paths

N_GRAPHS = 30
TRAIN_MODE = False

#--- Train ---
if TRAIN_MODE:
    graph_path = paths.graph_path_train
    dataset_path = paths.dataset_path_train
#--- Dev ---
else:
    graph_path = paths.graph_path_dev
    dataset_path = paths.dataset_path_dev

with open(dataset_path, "r") as read_file:
    dataset = json.load(read_file)
read_file.close()

try:
    with open(graph_path, 'rb') as f:
        graphs_list = pickle.load(f)
    f.close()
except:
    graphs_list = list()

old_graphs_list_length = len(graphs_list)

for i in range(N_GRAPHS):

    print(f"--- SAMPLE {i} ---")

    sample = dataset[random.randint(0, len(dataset)-1)]
    sampleId = sample['id']
    
    
    try:
        graph, id2sentence = cg.buildCoreferenceGraph(sample['query'], sample['supports'])
    except:
        print("\n!!! ERROR in Building This Graph !!!\n")
        continue
    
    #graph, id2sentence = cg.buildCoreferenceGraph(sample['query'], sample['supports'])
    #print("\n--- Graph: ---\n")
    #print(graph.getEdges())

    answer_positions = list()
    print(f"sampleId: {sampleId}")
    print(f"Number of nodes: {len(graph.getNodes())}")

    for node in graph.getNodes():
        if node != 'q':
            sentence = id2sentence[node]
            if cg.shareAllWordsOfFirst(sample['answer'], sentence):
                answer_positions.append(node)
    print(f"answer_positions: {answer_positions}")
    if TRAIN_MODE:
        if len(answer_positions) == 0:
            continue


    elem = {'id':sampleId, 'graph':graph, 'id2sentence':id2sentence, 'answer_positions':answer_positions}
    graphs_list.append(elem)

    print(f"{len(graphs_list)}/{N_GRAPHS+old_graphs_list_length}")

    with open(graph_path, 'wb') as f:
        pickle.dump(graphs_list, f)
    f.close()

with open(graph_path, 'rb') as f:
    graphs_list = list(pickle.load(f))
    print(graphs_list)
f.close()