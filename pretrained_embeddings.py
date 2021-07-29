import paths
#from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor
import json
import dill
import pickle
import pandas as pd
import numpy as np
from keras import preprocessing

embedding_matrix_path = "embedding_matrix.pkl"

def create_embedding_matrix(word_index,embedding_dict,dimension):
    embedding_matrix=np.zeros((len(word_index)+1,dimension))
 
    for word, index in word_index.items():
        if word in embedding_dict:
            embedding_matrix[index]=embedding_dict[word]
    return embedding_matrix


glove = pd.read_csv(paths.embedding, sep=" ", quoting=3, header=None, index_col=0)
glove_embedding = {key: val.values for key, val in glove.T.items()}

with open("E:/Datasets/Wikihop/train.json", "r") as read_file:
    dataset = json.load(read_file)

#text=["The cat sat on mat","we can play with model"]

text = list()
for item in dataset:
    text.append(item["query"])
    text.append(item["answer"])
    text.append(item["supports"])

tokenizer = preprocessing.text.Tokenizer(split=" ", num_words=50000, oov_token='OOV')
tokenizer.fit_on_texts(text)
 
text_token=tokenizer.texts_to_sequences(text)
print(len(text_token))

#encoder = StaticTokenizerEncoder(text, min_occurrences=1, tokenize=lambda s: s.split())
#print(encoder.token_to_index)

print(len(tokenizer.word_index))
 
embedding_matrix=create_embedding_matrix(tokenizer.word_index, embedding_dict=glove_embedding,dimension=50)

print(embedding_matrix.shape)

file = open("tokenizer_glove.pkl", "wb")
dill.dump(tokenizer, file)
file.close()

with open(embedding_matrix_path, 'wb') as f:
        pickle.dump(embedding_matrix, f)
