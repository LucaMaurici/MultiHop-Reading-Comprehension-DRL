import pickle
import CoreferenceGraph as cg
import math
import matplotlib.pyplot as plt

graph_path = "CoreferenceGraphsList_dev_uniqueAnswerFiltered.pkl"
#graph_path = "CoreferenceGraphsList_train3_uniqueAnswerFiltered.pkl"

try:
    with open(graph_path, 'rb') as f:
        graphs_list = pickle.load(f)
    f.close()
except:
    graphs_list = list()

graphs_list_length = len(graphs_list)
print(f"len(graphs_list) = {graphs_list_length}")

unreachable_answers_counter = 0
total_number_of_hops = 0
total_number_of_hops_with_unreachable_answers = 0

total_degree = 0
total_number_of_nodes = 0

num_walks_more_than_10_hops = 0
num_walks_more_than_10_hops_with_unreachable_answers = 0

MEAN_DEGREE = 5.385
total_squared_deviation_degree = 0
degree2occ = {}

for i, elem in enumerate(graphs_list):
    print(f"\n--- SAMPLE {i} ---")

    hops_counter = 0

    graph = elem["graph"]
    id2sentence = elem["id2sentence"]

    nodes = graph.getNodes()
    nodes_len = len(nodes)
    print(f"Number of nodes: {nodes_len}")
    total_number_of_nodes += nodes_len
    for node in nodes:
        graph.setCurrentNode(node)
        node_degree = len(graph.getAdjacentNodes())
        total_degree += node_degree
        #print(node_degree)
        try:
            degree2occ[node_degree] += 1
        except:
            degree2occ[node_degree] = 1
        total_squared_deviation_degree += pow((node_degree - MEAN_DEGREE), 2)

    done = False

    graph.setCurrentNode('q')
    markers = ['q']

    actions = graph.getAdjacentNodes()
    #num_to_visit = 0
    num_iterations = 0
    to_visit = list()
    to_visit.append(1)
    to_visit += actions
    #num_to_visit += len(actions)+1

    for j, node in enumerate(to_visit):
        num_iterations += 1
        if type(node) == int:
            hops_counter = node
            continue
        if node in markers:
            continue
        

        markers.append(node)
        graph.setCurrentNode(node)

        if graph.currentNode == elem["answer_positions"][0]:
            done = True
            break

        actions = graph.getAdjacentNodes()
        to_visit.append(hops_counter+1)
        to_visit += actions
        #num_to_visit += len(actions)+1
    #print(f"num_to_visit: {num_to_visit}")
    print(f"len_to_visit: {len(to_visit)}")
    print(f"len_markers: {len(markers)}")
    print(f"len_markers_set: {len(set(markers))}")
    total_number_of_hops_with_unreachable_answers += hops_counter

    if hops_counter > 10:
            num_walks_more_than_10_hops_with_unreachable_answers += 1

    if done == False:
        print("UNREACHABLE ANSWER: 'done = False'")
        print(f"Number of hops: {hops_counter}")
        unreachable_answers_counter += 1
    else:
        print(f"Number of hops: {hops_counter}")
        total_number_of_hops += hops_counter
        average_number_of_hops = total_number_of_hops/((i+1)-unreachable_answers_counter)
        print(f"Average number of hops: {average_number_of_hops}")
        average_number_of_hops_unreachable_answers = total_number_of_hops_with_unreachable_answers/(i+1)
        print(f"Average number of hops including unreachable answers: {average_number_of_hops_unreachable_answers}")

        if hops_counter > 10:
            num_walks_more_than_10_hops += 1

average_number_of_hops = total_number_of_hops/(graphs_list_length-unreachable_answers_counter)
print(f"\n\n\n Average number of hops: {average_number_of_hops}")
average_number_of_hops_unreachable_answers = total_number_of_hops_with_unreachable_answers/graphs_list_length
print(f" Average number of hops including unreachable answers: {average_number_of_hops_unreachable_answers}")

#print(f"\nNumber of unreachable_answers: {unreachable_answers_counter}")
print(f"\n Percentage of reachable answers: {100-(unreachable_answers_counter/graphs_list_length)*100} %")

print(f"\n Mean degree: {total_degree/total_number_of_nodes}")

print(f"\n Percentage of reachable answers in more than 10 hops: {(num_walks_more_than_10_hops/(graphs_list_length-unreachable_answers_counter))*100} %")
print(f" Percentage of answers in more than 10 hops including unreachable answers: {(num_walks_more_than_10_hops_with_unreachable_answers/(graphs_list_length-unreachable_answers_counter))*100} %")

print(f"\n Degree variance: {math.sqrt(total_squared_deviation_degree/(total_number_of_nodes-1))}")

plt.bar(list(degree2occ.keys()), degree2occ.values(), color='g')
plt.show()

print()