import pickle
import networkx as nx
import matplotlib.pyplot as plt

import MultiHopEnvironment
import paths
#import random
import math
import random
import json
import utils

import myPredictor

USE_CANDIDATES = True

graph_path = "CoreferenceGraphsList_dev3.pkl"

try:
    with open(graph_path, 'rb') as f:
        graphs_list = pickle.load(f)
    f.close()
except:
    graphs_list = list()
n_samples = len(graphs_list)

with open(paths.dataset_path_dev,"r") as read_file:
    dataset = json.load(read_file)

em_score_tot = 0

for i, graphs_list_item in enumerate(graphs_list):

    sampleId = graphs_list_item["id"]
    graph = graphs_list_item["graph"]
    id2sentence = graphs_list_item["id2sentence"]
    sample = MultiHopEnvironment.getSampleById(dataset, sampleId)
    answer = sample["answer"]
    candidates = sample["candidates"]

    G = nx.DiGraph()
    G.add_edges_from(graph.getEdges())
    shortest_paths = nx.single_source_shortest_path(G, source='q', cutoff=6)
    #print(len(shortest_paths))
    sequence = list()
    for value in shortest_paths.values():
        sequence += value
    sequence = list(set(sequence))
    sequence.remove('q')
    text_to_read = ""
    for item in sequence:
        text_to_read += id2sentence[item]
    #print(text_to_read)
    if USE_CANDIDATES: prediction = myPredictor.myPredict(text_to_read, id2sentence['q'], candidates=candidates)
    else: prediction = myPredictor.myPredict(text_to_read, id2sentence['q'])

    print(f"\nPrediction: {prediction}")
    print(f"Correct answer: {answer}")

    if len(prediction) > 0:
        prediction = prediction[0][0]
    else:
        prediction = ""
    em_score_sample = int(utils.exact_match_score(prediction, answer))
    print(f"EM Score - Sample: {em_score_sample*100}%")

    em_score_tot += em_score_sample
    print(f"EM Score - partial: {(em_score_tot/(i+1))*100}%")
    print("------------------------------------------------------------------------\n\n")

print(f"EM Score - Final: {(em_score_tot/n_samples)*100}%")

'''
    #candidates
    #sample = getSampleById(dataset, sampleId)
    print(candidates)

    if text_to_read != "":
        #prediction = myPredictor.myPredict(text_to_read, question, candidates=candidates)
        prediction = myPredictor.myPredict(text_to_read, question)
    else:
        prediction = [("", 1.0)]
    
    print(f"\nPrediction: {prediction}")
    print(f"Correct answer: {answer}")

    if len(prediction) > 0:
        prediction = prediction[0][0]
    else:
        prediction = ""
    em_score_sample = int(utils.exact_match_score(prediction, answer))
    print(f"EM Score - Sample: {em_score_sample*100}%")

    em_score_tot += em_score_sample
    print(f"EM Score - partial: {(em_score_tot/(i+1))*100}%")
    print("------------------------------------------------------------------------\n\n")

print(f"EM Score - Final: {(em_score_tot/n_samples)*100}%")
'''