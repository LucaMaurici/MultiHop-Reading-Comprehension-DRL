import pickle
import CoreferenceGraph as cg
import random

readFromFile = True

if readFromFile:
	file = open("CoreferenceGraph.pkl", "rb")
	graph = pickle.load(file)
	file = open("id2sentence.pkl", "rb")
	id2sentence = pickle.load(file)
else:
	graph, id2sentence = cg.buildCoreferenceGraph()

print("\n--- Graph: ---\n")
print(graph.getEdges())

print("\n--- Adjacent nodes do the first one: ---\n")
graph.setCurrentNode('d0s0')
print(graph.currentNode)
print(graph.getAdjacentNodes())

print("\n--- Random walk: ---\n")
walkIDs = [graph.currentNode]
walkSentences = [id2sentence[graph.currentNode]]
for i in range(0,500):
	print("I: ",i)
	adjacentNodes = graph.getAdjacentNodes()
	print(adjacentNodes)
	adjacentNodesLength = len(adjacentNodes)
	if adjacentNodesLength != 0:
		randomIndex = random.randint(0, adjacentNodesLength-1)
	else:
		break
	print(randomIndex)
	graph.goTo(adjacentNodes[randomIndex])
	walkIDs.append(graph.currentNode)
	walkSentences.append(id2sentence[graph.currentNode])
	print(graph.currentNode)

print("\n--- Result: ---\n")
for sID, sentence in zip(walkIDs, walkSentences):
	print(sID, ': ', sentence)