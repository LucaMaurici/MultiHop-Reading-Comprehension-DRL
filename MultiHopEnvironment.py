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
        
        self.graph, self.id2sentence = cg.buildCoreferenceGraph(sample['query'], sample['supports'])
        print("\n--- Graph: ---\n")
        print(self.graph.getEdges())

        print("\n--- Adjacent nodes do the first one: ---\n")
        self.graph.setCurrentNode('q')
        print(self.graph.currentNode)
        print(self.graph.getAdjacentNodes())

        
        self.actions = self.graph.getAdjacentNodes()
        self.state.append(list())
        for actionID in actions:
            self.state[1].append(id2sentence[actionID])

        self.state.append(list())
        #self.done = False

        return state

    def step(self, actionIndex):
        reward = 0
        done = False

        if actionIndex < len(actions):
            self.graph.goTo(self.actions[actionIndex])
        else:
            reward = -0.1

        if cg.shareWords(state[0], id2sentence[self.graph.currentNode]):
            done = True
            reward = 1

        return self.id2sentence[self.graph.currentNode], reward, done