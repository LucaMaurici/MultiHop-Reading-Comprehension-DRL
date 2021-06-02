import pickle
import CoreferenceGraph as cg
import json
import random

N_GRAPHS = 10
TRAIN_MODE = True
file_name = 'CoreferenceGraphsListTrain.pkl'
#---Train---
dataset_path = "C:/Users/corri/Desktop/RLProjects/MultiHop-Reading-Comprehension-DRL/dataset/Wikihop/train.json"
#dataset_path = "E:/Datasets/Wikihop/train.json"
#---Test---
#dataset_path = "C:/Users/corri/Desktop/RLProjects/MultiHop-Reading-Comprehension-DRL/dataset/Wikihop/dev.json"
#dataset_path = "E:/Datasets/Wikihop/dev.json"

with open(dataset_path, "r") as read_file:
    dataset = json.load(read_file)
read_file.close()

try:
    with open(file_name, 'rb') as f:
        graphs_list = pickle.load(f)
    f.close()
except:
    graphs_list = list()

for i in range(N_GRAPHS):

    print(f"--- SAMPLE {i} ---")

    sample = dataset[random.randint(0, len(dataset)-1)]
    sampleId = sample['id']
    
    
    try:
        graph, id2sentence = cg.buildCoreferenceGraph(sample['query'], sample['supports'])
    except:
        print("\n!!! ERROR in building a graph !!!\n")
        continue
    
    #graph, id2sentence = cg.buildCoreferenceGraph(sample['query'], sample['supports'])
    #print("\n--- Graph: ---\n")
    #print(graph.getEdges())

    if TRAIN_MODE:
        answer_positions = list()
        print(f"sampleId: {sampleId}")
        print(f"Number of nodes: {len(graph.getNodes())}")

        for node in graph.getNodes():
            if node != 'q':
                sentence = id2sentence[node]
                if cg.shareAllWordsOfFirst(sample['answer'], sentence):
                    answer_positions.append(node)
                print(f"answer_positions: {answer_positions}")
        if len(answer_positions) == 0:
            continue


    elem = {'id':sampleId, 'graph':graph, 'id2sentence':id2sentence, 'answer_positions':answer_positions}
    graphs_list.append(elem)

    print(f"{len(graphs_list)}/{N_GRAPHS}")

    with open(file_name, 'wb') as f:
        pickle.dump(graphs_list, f)
    f.close()

with open(file_name, 'rb') as f:
    graphs_list = list(pickle.load(f))
    print(graphs_list)
f.close()