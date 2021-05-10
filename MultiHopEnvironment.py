import CoreferenceGraph as cg
import json
import random
import pickle
import Graph_class as gs
import dill
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor
import numpy as np

def getSampleById(dataset, id):
    for e in dataset:
        if e['id'] == id:
            return e
    return None

'''
def encodeState(state, encoder):
    encodedState = list()
    for i, e in enumerate(state):
        if i==0:
            encodedState.append(encoder.encode(e).numpy())
        else:
            encodedState.append(list())
            for sentence in e:
                encodedState[i].append(encoder.encode(sentence).numpy())
            encodedState[i] = np.array(encodedState[i])
    print("\n---Numpy---")
    print(np.array(encodedState))
    return np.array(encodedState)
'''
'''
def encodeState(state, encoder):
    encodedState = list()
    for i, e in enumerate(state):
        if i==0:
            encodedState.append(encoder.encode(e).numpy())
        else:
            for sentence in e:
                encodedState.append(encoder.encode(sentence).numpy())
            encodedState = np.array(encodedState)
    print("\n---Numpy---")
    print(np.array(encodedState))
    return np.array(encodedState)
'''

def padState(state):
    MAX_LEN = 50
    newState = list()
    print(newState)
    for i, e in enumerate(state):
        print(e)
        trueLen = len(e)
        if trueLen > 50:
            newState.append(e[:50])
        else:
            for i in range (50 - trueLen):
                e.append(0)
            newState.append(e)
            print(newState)
    print(newState)
    return newState

def encodeState(state, encoder):
    print(state)
    encodedState = list()
    for i, e in enumerate(state):
        if i==0:
            encodedState.append(encoder.encode(e).tolist())
        elif i==1:
            num = 0
            for sentence in e:
                encodedState.append(encoder.encode(sentence).tolist())
                num +=1
            for j in range(8-num):
                encodedState.append(list())
        elif i==2:
            num = 0
            for sentence in e:
                encodedState.append(encoder.encode(sentence).tolist())
                num +=1
            for j in range(30-num):
                encodedState.append(list())
    print("\n---Numpy---")
    print(encodedState)
    encodedState = padState(encodedState)
    return encodedState

class MultiHopEnvironment:

    def __init__(self):
        #with open("./Dataset/train.json", "r") as read_file:
        with open("E:/Datasets/Wikihop/train.json", "r") as read_file:
            self.dataset = json.load(read_file)
        #self.reset()
        with open('CoreferenceGraphsList.pkl', 'rb') as f:
            self.graphs_list = pickle.load(f)

        with open('StaticTokenizerEncoder.pkl', 'rb') as f:
            self.encoder = dill.load(f)
        print('Encoder opened')
        

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
        self.state = [sample['query']] #  state[0] = query
        self.answer = sample['answer']


        print("\n--- Adjacent nodes do the first one: ---\n")
        self.graph.setCurrentNode('q')
        print(self.graph.currentNode)
        print(self.graph.getAdjacentNodes())

        
        self.actions = self.graph.getAdjacentNodes()
        self.state.append(list()) #  state[1] = sentences representing the possible actions
        for actionID in self.actions:
            self.state[1].append(self.id2sentence[actionID])

        self.state.append(list()) #  state[2] = sentences representing the history of my hopes, initially empty
        #self.done = False

        output = encodeState(self.state, self.encoder)
        print("\n---STATE DOPO RESET---")
        #print(output)
        return output


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

        return np.array(encodeState(self.state, self.encoder), dtype=object), reward, done


#  state[0] = query
#  state[1] = 