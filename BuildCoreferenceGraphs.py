import pickle
import CoreferenceGraph as cg
import json
import random

N_GRAPHS = 15

with open("./Dataset/train.json", "r") as read_file:
    dataset = json.load(read_file)

try:
    with open('CoreferenceGraphsList.pkl', 'rb') as f:
        graphs_list = pickle.load(f)
except:
    graphs_list = list()

for i in range(N_GRAPHS):

    sample = dataset[random.randint(0, len(dataset)-1)]
    sampleId = sample['id']
    
    try:
        graph, id2sentence, id2modSentence = cg.buildCoreferenceGraph(sample['query'], sample['supports'])
    except:
        continue
    print("\n--- Graph: ---\n")
    print(graph.getEdges())

    elem = {'id':sampleId, 'graph':graph, 'id2sentence':id2sentence, 'id2modSentence':id2modSentence}
    graphs_list.append(elem)

    with open('CoreferenceGraphsList.pkl', 'wb') as f:
        pickle.dump(graphs_list, f)

with open('CoreferenceGraphsList.pkl', 'rb') as f:
    graphs_list = list(pickle.load(f))
    print(graphs_list)