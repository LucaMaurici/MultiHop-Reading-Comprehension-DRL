from stanza.server import CoreNLPClient


import stanza

import time

corenlp_dir = './corenlp'
stanza.install_corenlp(dir=corenlp_dir)

# Set the CORENLP_HOME environment variable to point to the installation location
import os
os.environ["CORENLP_HOME"] = "./corenlp"



# start a CoreNLP client
#with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner','parse','coref']) as client:
with CoreNLPClient(annotators=['tokenize']) as client:
    
    # run annotation over input
    ann = client.annotate('')
    time.sleep(172800) #  2 days
    ann = client.annotate('')
    #exit()
