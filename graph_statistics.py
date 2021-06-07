import pickle
import CoreferenceGraph as cg
#import paths

graph_path = "CoreferenceGraphsList_dev_uniqueAnswerFiltered.pkl"

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

for i, elem in enumerate(graphs_list):
    print(f"\n--- SAMPLE {i} ---")

    hops_counter = 0

    graph = elem["graph"]
    id2sentence = elem["id2sentence"]

    nodes = graph.getNodes()
    total_number_of_nodes += len(nodes)
    for node in nodes:
        graph.setCurrentNode(node)
        total_degree += len(graph.getAdjacentNodes())

    done = False

    graph.setCurrentNode('q')
    markers = ['q']

    actions = graph.getAdjacentNodes()
    to_visit = list()
    to_visit.append(1)
    to_visit += actions
    

    for node in to_visit:
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
print(f"\n\nAverage number of hops: {average_number_of_hops}")
average_number_of_hops_unreachable_answers = total_number_of_hops_with_unreachable_answers/graphs_list_length
print(f"Average number of hops including unreachable answers: {average_number_of_hops_unreachable_answers}")

print(f"\nNumber of unreachable_answers: {unreachable_answers_counter}")
print(f"\nPercentage of reachable answers: {100-(unreachable_answers_counter/graphs_list_length)*100} %")

print(f"\nMean weighted degree: {total_degree/total_number_of_nodes}")

print(f"\nPercentage of reachable answers in more than 10 hops: {(num_walks_more_than_10_hops/(graphs_list_length-unreachable_answers_counter))*100} %")
print(f"\nPercentage of reachable answers in more than 10 hops: {(num_walks_more_than_10_hops_with_unreachable_answers/(graphs_list_length-unreachable_answers_counter))*100} %")

print()