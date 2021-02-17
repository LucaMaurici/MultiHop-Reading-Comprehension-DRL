import CoreferenceGraph as cg
import json
import random
import pickle
import Graph_class as gs

def getSampleById(dataset, id):
    for e in dataset:
        if e['id'] == id:
            return e
    return None

class MultiHopEnvironment:

    def __init__(self):
        with open("./Dataset/train.json", "r") as read_file:
            self.dataset = json.load(read_file)
        #self.reset()
        with open('CoreferenceGraphsList.pkl', 'rb') as f:
            self.graphs_list = pickle.load(f)
        

    def reset(self):
        '''
        sample = self.dataset[random.randint(0, len(self.dataset)-1)]
        self.state = [sample['query']]
        self.answer = sample['answer']
        
        self.graph, self.id2sentence = cg.buildCoreferenceGraph(sample['query'], sample['supports'])
        print("\n--- Graph: ---\n")
        print(self.graph.getEdges())
        '''

        graphSample = self.graphs_list[random.randint(0, len(self.graphs_list)-1)]
        self.sampleId = graphSample['id']
        self.graph = graphSample['graph']
        self.id2sentence = graphSample['id2sentence']

        #print("\n--- Graph: ---\n")
        #print(self.graph.getEdges())

        sample = getSampleById(self.dataset, self.sampleId)
        self.state = [sample['query']]
        self.answer = sample['answer']


        print("\n--- Adjacent nodes do the first one: ---\n")
        self.graph.setCurrentNode('q')
        print(self.graph.currentNode)
        print(self.graph.getAdjacentNodes())

        
        self.actions = self.graph.getAdjacentNodes()
        self.state.append(list())
        for actionID in self.actions:
            self.state[1].append(self.id2sentence[actionID])

        self.state.append(list())
        #self.done = False

        return self.state


    def step(self, actionIndex):
        reward = 0
        done = False

        if actionIndex < len(self.actions):
            self.graph.goTo(self.actions[actionIndex])
        else:
            reward = -0.1
            return self.state, reward, done

        if cg.shareWords(self.answer, self.id2sentence[self.graph.currentNode]):
            done = True
            reward = 1

        self.actions = self.graph.getAdjacentNodes()
        self.state[1] = []
        for actionID in self.actions:
            self.state[1].append(self.id2sentence[actionID])
        self.state[2].append(self.id2sentence[self.graph.currentNode])

        return self.state, reward, done