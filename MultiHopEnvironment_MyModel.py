import CoreferenceGraph as cg
import json
import random
import pickle
import Graph_class as gs
import dill

from keras import preprocessing

import numpy as np
import paths

NUM_ACTIONS = 31

def getSampleById(dataset, id):
    for e in dataset:
        if e['id'] == id:
            return e
    return None

def padState(state):
    MAX_LEN = 50
    newState = list()
    for i, e in enumerate(state):
        trueLen = len(e)
        if trueLen > 50:
            newState.append(e[:50])
        else:
            for i in range (50 - trueLen):
                e.append(0)
            newState.append(e)
    return newState

def encodeState(state, encoder):
    encodedState = list()
    for i, e in enumerate(state):
        if i==0:
            #encodedState.append(encoder.encode(e).tolist())
            #print(encoder.texts_to_sequences([e])[0])
            encodedState.append(encoder.texts_to_sequences([e])[0])
        elif i==1:
            num = 0
            if len(e) > NUM_ACTIONS:  # truncate actions if num_actions > NUM_ACTIONS
                e = e[0:NUM_ACTIONS]
            for sentence in e:
                #encodedState.append(encoder.encode(sentence).tolist())
                encodedState.append(encoder.texts_to_sequences([sentence])[0])
                num +=1
            for j in range(NUM_ACTIONS-num):  # pad if num_actions < NUM_ACTIONS
                encodedState.append(list())
        elif i==2:
            num = 0
            for sentence in e:
                #encodedState.append(encoder.encode(sentence).tolist())
                encodedState.append(encoder.texts_to_sequences([sentence])[0])
                num +=1
            for j in range(30-num):
                encodedState.append(list())

    #print("\n---Numpy---")
    #print(type(encodedState[0][0]))
    encodedState = padState(encodedState)
    return encodedState


class MultiHopEnvironment_MyModel:

    def __init__(self, train_mode = True):
        self.train_mode = train_mode
        self.test_index = 0
        if(train_mode):
            graph_path = paths.graph_path_train
            dataset_path = paths.dataset_path_train
        else:
            graph_path = paths.graph_path_dev
            dataset_path = paths.dataset_path_dev

        with open(dataset_path,"r") as read_file:
            self.dataset = json.load(read_file)

        with open(graph_path, 'rb') as f:
            self.graphs_list = pickle.load(f)

        with open('tokenizer_glove.pkl', 'rb') as f:
            self.encoder = dill.load(f)

    def reset(self):
        if self.train_mode == False:
            if self.test_index < len(self.graphs_list):
                graphSample = self.graphs_list[self.test_index]
                self.test_index += 1
            else:
                print("\nDEV_DATASET FINISHED\n")
                exit()
        else:
            graphSample = self.graphs_list[random.randint(0, len(self.graphs_list)-1)]
        self.sampleId = graphSample['id']
        self.graph = graphSample['graph']
        self.id2sentence = graphSample['id2sentence']
        self.answer_positions = graphSample['answer_positions']
        self.node2value = graphSample['node2value']

        sample = getSampleById(self.dataset, self.sampleId)

        self.state = [sample['query']] #  state[0] = query
        self.answer = sample['answer']

        self.graph.setCurrentNode('q')
        
        self.actions = self.graph.getAdjacentNodes()
        self.state.append(list()) #  state[1] = sentences representing the possible actions
        for actionID in self.actions:
            self.state[1].append(self.id2sentence[actionID])

        self.state.append(list()) #  state[2] = sentences representing the history of my hopes, initially empty

        encoded_state = encodeState(self.state, self.encoder)

        return encoded_state, self.state, self.answer, sample['candidates'], self.node2value[self.actions]


    def step(self, actionIndex):
        reward = -1
        done = False

        if actionIndex < len(self.actions):
            self.graph.goTo(self.actions[actionIndex])
        else:
            reward = -1.1
            self.graph.goTo(self.actions[random.randint(0, len(self.actions)-1)])
            #return np.array(encodeState(self.state, self.encoder)), reward, done, self.state

        self.actions = self.graph.getAdjacentNodes()
        self.state[1] = []
        for actionID in self.actions:
            self.state[1].append(self.id2sentence[actionID])
        self.state[2].append(self.id2sentence[self.graph.currentNode])

        #if cg.shareWords(self.answer, self.id2sentence[self.graph.currentNode]):
        if self.graph.currentNode in self.answer_positions:
            done = True
            reward = 10

        return np.array(encodeState(self.state, self.encoder)), reward, done, self.state, self.node2value[self.actions]
