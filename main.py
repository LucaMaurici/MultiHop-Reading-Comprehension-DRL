import pickle
import CoreferenceGraph as cg
import random

readFromFile = True

if readFromFile:
	file = open("CoreferenceGraph.pkl", "rb")
	graph = pickle.load(file)
else:
	graph = cg.buildCoreferenceGraph()

print("\n--- Graph: ---\n")
print(graph.getEdges())

print("\n--- Adjacent nodes do the first one: ---\n")
graph.setCurrentNode('d0s0')
print(graph.currentNode)
print(graph.getAdjacentNodes())

print("\n--- Random walk: ---\n")
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
	print(graph.currentNode)

#print("\n--- Policy walk: ---\n")