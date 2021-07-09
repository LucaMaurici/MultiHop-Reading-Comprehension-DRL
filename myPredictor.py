import torch
from predictor import Predictor
from multiprocessing import cpu_count, freeze_support
import sys
#print("\nPATH: ")
#print(f"{sys.path}\n\n")

predictor = Predictor(
    #"data/models/m_reader.mdl",
    "data/models/m_reader_retrained2.4.2.mdl.checkpoint",
    #"C:\\Users\\lucam\\Documents\\GitHub\\Personale\\MultiHop-Reading-Comprehension-DRL\\MnemonicReader\\data\\models\\m_reader.mdl",
    normalize=True,
    #embedding_file=None,
    #char_embedding_file=None,
    #num_workers=int(cpu_count()/2),
    num_workers=0,
)

def myPredict(document, question, candidates=None, top_n=1):
    """Predict a single document - question pair."""
    result = predictor.predict(document, question, candidates, top_n)
    return result
