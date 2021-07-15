from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor
import pickle
import json
import dill

def getSampleById(dataset, id):
    for e in dataset:
        if e['id'] == id:
            return e
    return None

print('Start')

with open("E:/Datasets/Wikihop/train.json", "r") as read_file:
    dataset = json.load(read_file)

print('Dataset loaded')

with open('CoreferenceGraphsList.pkl', 'rb') as f:
    graphs_list = pickle.load(f)

print('GraphList loaded')

loaded_data = list()
for graphSample in graphs_list:
    sampleId = graphSample['id']
    elem = getSampleById(dataset, sampleId)
    loaded_data.append(elem['query'])
    for document in elem['supports']:
        loaded_data.append(document)


print('Data loaded')

encoder = StaticTokenizerEncoder(loaded_data, min_occurrences=1, tokenize=lambda s: s.split())

print('Encoder fitted')

file = open("StaticTokenizerEncoder.pkl", "wb")
dill.dump(encoder, file)
file.close()

print('Encoder saved')
encoder = None
with open('StaticTokenizerEncoder.pkl', 'rb') as f:
    encoder = dill.load(f)

print('Encoder opened')

#example_data = ['ehi, ehi ciao', 'funny', 'paper', 'ciao', 'hello', 'the', 'pen', 'is', 'on', 'the', 'table', 'wikipedia', 'The']
#encoded_data = [encoder.encode(example) for example in example_data]
#print(encoded_data)
print(encoder.vocab)

print(len(encoder.vocab))

print(encoder.encode("Oparara Valerio"))