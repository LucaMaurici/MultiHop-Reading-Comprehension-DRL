import pickle
import CoreferenceGraph as cg
import json
import random

N_GRAPHS = 960

with open("E:/Datasets/Wikihop/train.json", "r") as read_file:
    dataset = json.load(read_file)
read_file.close()

try:
    with open('CoreferenceGraphsList.pkl', 'rb') as f:
        graphs_list = pickle.load(f)
    f.close()
except:
    graphs_list = list()

for i in range(N_GRAPHS):

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

    elem = {'id':sampleId, 'graph':graph, 'id2sentence':id2sentence}
    graphs_list.append(elem)

    with open('CoreferenceGraphsList.pkl', 'wb') as f:
        pickle.dump(graphs_list, f)
    f.close()

with open('CoreferenceGraphsList.pkl', 'rb') as f:
    graphs_list = list(pickle.load(f))
    print(graphs_list)
f.close()