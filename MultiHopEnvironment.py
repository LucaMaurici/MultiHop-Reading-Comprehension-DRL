import CoreferenceGraph as cg
import json
import random

class MultiHopEnvironment:

    def __init__(self):
        with open("./Dataset/train.json", "r") as read_file:
            self.dataset = json.load(read_file)
        #self.reset()
    
    def reset(self):
        sample = self.dataset[random.randint(0, len(self.dataset)-1)]
        self.state = [sample['query']]
        
        graph, id2sentence = cg.buildCoreferenceGraph(sample['query'], sample['supports'])
        print("\n--- Graph: ---\n")
        print(graph.getEdges())

        print("\n--- Adjacent nodes do the first one: ---\n")
        graph.setCurrentNode('d0s0')
        print(graph.currentNode)
        print(graph.getAdjacentNodes())

        
        self.actions = []
        