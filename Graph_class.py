class Graph:

    currentNode = None

    def __init__(self, node2Edges=None):
        if node2Edges is None:
            node2Edges = {}
        self.node2Edges = node2Edges


    # Get the keys of the dictionary
    def getNodes(self):
        return list(self.node2Edges.keys())

    # Find the distinct list of edges
    def getEdges(self):
        edges = []
        for node in self.node2Edges:
            for nextNode in self.node2Edges[node]:
                if [node, nextNode] not in edges:
                    edges.append([node, nextNode])
        return edges


    # Add the node as a key
    def addNode(self, node):
       if node not in self.node2Edges:
            self.node2Edges[node] = []

    # Add the new edge
    def addEdge(self, edge):
        #edge = set(edge)
        (node1, node2) = tuple(edge)
        if node1 in self.node2Edges:
            self.node2Edges[node1].append(node2)
            self.node2Edges[node1] = list(set(self.node2Edges[node1]))  # remove duplicates
        else:
            self.node2Edges[node1] = [node2]

    
    def setCurrentNode(self, node):
        self.currentNode = node

    def getAdjacentNodes(self):
        #if currentNode != None:
        return self.node2Edges[self.currentNode]
        #return []

    def goTo(self, node):
        self.currentNode = node
