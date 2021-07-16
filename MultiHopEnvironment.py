import CoreferenceGraph as cg
import json
import random
import pickle
import Graph_class as gs
import dill

#from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor
from keras import preprocessing

import numpy as np
import paths

NUM_ACTIONS = 20

def getSampleById(dataset, id):
    for e in dataset:
        if e['id'] == id:
            return e
    return None

'''
def getSampleById_squad(dataset, id):
    id = "WikiHop_q_" + id
    for e in dataset['data'][0]['paragraphs']:
        if e['qas'][0]['id'] == id:
            return e
    return None
'''

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

# TODO ma mi sa non utile
def unpad_state(state):
    return state

# non si usa mai
def decode_state(state, encoder):
    decoded_state = list()
    actions = list()
    history = list()
    for i, e in enumerate(state):
        if i==0:
            #question = encoder.decode(e)
            question = encoder.sequences_to_texts(e)
            decoded_state.append(question)
        elif i>=1 and i<=NUM_ACTIONS:
            #actions.append(encoder.decode(e))
            actions.append(encoder.sequences_to_texts(e))
        elif i>=NUM_ACTIONS+1 <=30+NUM_ACTIONS:
            #history.append(encoder.decode(e))
            history.append(encoder.sequences_to_texts(e))
    decoded_state.append(actions)
    decoded_state.append(history)
    decoded_state = unpad_state(decoded_state)
    print(decoded_state)
    return decoded_state


class MultiHopEnvironment:

    def __init__(self, train_mode = False):
        self.train_mode = train_mode
        self.test_index = 0
        if(train_mode):
            graph_path = paths.graph_path_train
            dataset_path = paths.dataset_path_train
            #dataset_path_squad = paths.dataset_path_train_squad
        else:
            graph_path = paths.graph_path_dev
            dataset_path = paths.dataset_path_dev
            #dataset_path_squad = paths.dataset_path_dev_squad

        #with open("./Dataset/train.json", "r") as read_file:
        with open(dataset_path,"r") as read_file:
            self.dataset = json.load(read_file)
        '''
        with open(dataset_path_squad,"r") as read_file:
            self.dataset_squad = json.load(read_file)
        '''
        #self.reset()
        with open(graph_path, 'rb') as f:
            self.graphs_list = pickle.load(f)

        '''
        with open('StaticTokenizerEncoder.pkl', 'rb') as f:
            self.encoder = dill.load(f)
        '''
        with open('tokenizer_glove.pkl', 'rb') as f:
            self.encoder = dill.load(f)
        #print('Encoder opened')
        

    def reset(self):
        '''
        sample = self.dataset[random.randint(0, len(self.dataset)-1)]
        self.state = [sample['query']]
        self.answer = sample['answer']
        
        self.graph, self.id2sentence = cg.buildCoreferenceGraph(sample['query'], sample['supports'])
        print("\n--- Graph: ---\n")
        print(self.graph.getEdges())
        '''
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

        #print("\n--- Graph: ---\n")
        #print(self.graph.getEdges())

        sample = getSampleById(self.dataset, self.sampleId)
        '''
        print("\n\n\n---SAMPLE 1---")
        print(sample)
        sample_squad = getSampleById_squad(self.dataset_squad, self.sampleId)
        print("\n---SAMPLE 2---")
        print(sample_squad)
        print("\n\n\n")
        '''
        self.state = [sample['query']] #  state[0] = query
        self.answer = sample['answer']


        #print("\n--- Adjacent nodes do the first one: ---\n")
        self.graph.setCurrentNode('q')
        #print(self.graph.currentNode)
        #print(self.graph.getAdjacentNodes())

        
        self.actions = self.graph.getAdjacentNodes()
        self.state.append(list()) #  state[1] = sentences representing the possible actions
        for actionID in self.actions:
            self.state[1].append(self.id2sentence[actionID])

        self.state.append(list()) #  state[2] = sentences representing the history of my hopes, initially empty
        #self.done = False

        encoded_state = encodeState(self.state, self.encoder)
        #print("\n---STATE DOPO RESET---")
        #print(output)
        return encoded_state, self.state, self.answer, sample['candidates']


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

        return np.array(encodeState(self.state, self.encoder)), reward, done, self.state


#  state[0] = query
#  state[1] = 