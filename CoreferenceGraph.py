from pynlp import StanfordCoreNLP
#from nltk import tokenize
#import nltk
#nltk.download('punkt')
import Graph_class as gs
import pickle

def indexer(docNumber, sentenceIndex):
    return 'd'+str(docNumber) + 's'+str(sentenceIndex)

def getRadixNode(node):
    dPos = node.index('s')
    docNumber = node[1:dPos]
    #print(f"Indexer: {indexer(docNumber, 0)}")
    return indexer(docNumber, 0)

def shareEntities(list1, list2):
    for e1 in list1:
        if e1 in list2:
            return True
    return False

'''
def shareWords(questOrAns, sentence2):
    print("\n✪✪✪ SHARE WORDS: ✪✪✪\n")
    for token in nlp(questOrAns.lower()):
        print(token, '->', token.pos)
        print(token, '->', token.lemma)
    list1 = questOrAns.lower().split()
    list2 = sentence2.lower().split()
    for e1 in list1:
        if e1 in list2:
            return True
    return False
'''
def keepMeaningfulWords(questOrAns):
    annotators = 'pos, lemma'
    nlp = StanfordCoreNLP(annotators=annotators)
    #print("\n✪✪✪ SHARE WORDS: ✪✪✪\n")
    questOrAns = questOrAns.lower()
    questOrAnsU = questOrAns.encode('utf-8')
    
    for token in nlp(questOrAnsU)[0]:
        #print(token, '->', token.pos)
        #print(token, '->', token.lemma)
        if token.pos == 'IN' or token.pos=='CC' or token.pos=='DT' or token.pos=='PRP' or token.pos=='PRP$' or token.pos=='TO' \
            or token.pos == 'WDT' or token.pos == 'WP' or token.pos == 'WP$' or token.pos == 'WRB':
            for char in ['.',';','?','!',':','"','\'']:
                questOrAns = questOrAns.replace(' '+str(token)+char, char)
            questOrAns = questOrAns.replace(' ' + str(token) + ' ', ' ')
            #print(questOrAns)
        if token.lemma=='be' or token.lemma=='have':
            questOrAns = questOrAns.replace(' ' + str(token) + ' ', ' ')
    #print("\n✪✪✪ SENTENCE: ✪✪✪\n")
    #print('QuestOrAns: ', questOrAns)
    return questOrAns

def shareWords(questOrAns, sentence2):
    questOrAns = keepMeaningfulWords(questOrAns)
    list1 = questOrAns.split()
    list2 = sentence2.lower().split()
    for e1 in list1:
        if e1 in list2:
            return True
    return False

def shareAllWordsOfFirst(questOrAns, sentence2):
    questOrAns = keepMeaningfulWords(questOrAns)
    list1 = questOrAns.split()
    list2 = sentence2.lower().split()
    for e1 in list1:
        if e1 not in list2:
            return False
    return True

def buildCoreferenceGraph(question, documents):

    annotators = 'tokenize, ssplit, pos, lemma, ner, entitymentions, coref, sentiment, openie'
    options = {'openie.resolve_coref': True}

    nlp = StanfordCoreNLP(annotators=annotators, options=options)


    temp = list()
    for document in documents:
        document = document.replace('%', ' percent')
        #document = document.replace('+', '<plus>')
        temp.append(document)
    documents = temp
    #print(documents)

    graph = gs.Graph()

    id2sentence = {}
    id2entities = {}
    #nodes2radix = {}

    graph.addNode('q')
    id2sentence['q'] = question

    num_documents = len(documents)

    for docNumber, text in enumerate(documents, start=0):

        #print('\n\n********* STARTING DOCUMENT ', docNumber, ' *********\n\n')

        document = nlp(text)
        
        #print("\n--- SENTENCE SPLITTING and AddToGraph: ---\n")
        for sentenceIndex, sentence in enumerate(document, start = 0):
            id2sentence[indexer(docNumber, sentenceIndex)] = str(sentence)
            graph.addNode(indexer(docNumber, sentenceIndex))
            if sentenceIndex != 0:
                graph.addEdge((indexer(docNumber, sentenceIndex-1), indexer(docNumber, sentenceIndex)))

        #LINKING BETWEEN THE LAST SENTENCE OF A DOCUMENT AND THE FIRST SENTENCE OF THE FOLLOWING DOCUMENT
        if docNumber != num_documents-1:
            graph.addEdge((indexer(docNumber, sentenceIndex), indexer(docNumber+1, 0)))
        #LINKING BETWEEN THE LAST SENTENCE OF THE LAST DOCUMENT AND THE FIRST SENTENCE OF FIRST DOCUMENT
        elif docNumber == num_documents-1:
            graph.addEdge((indexer(docNumber, sentenceIndex), indexer(0, 0)))

        #print("\n--- id2sentence: ---\n")
        #print(id2sentence)
        

        #print("\n--- Coreference resolution: ---\n")
        #for var in document.coref_chains:
            #print(var)



        #print("\n--- Coreference resolution chain: ---\n")
        list_of_listOfCorefSentenceIndexes = list()
        for i, chain in enumerate(document.coref_chains, start=0):
            list_of_listOfCorefSentenceIndexes.append(list())
            for mention in chain._coref.mention:
                list_of_listOfCorefSentenceIndexes[i].append(mention.sentenceIndex)


        #print()
        #print("Prima: ", list_of_listOfCorefSentenceIndexes)
        temp_list = list()
        for listElem in list_of_listOfCorefSentenceIndexes:
            temp_list.append(list(set(listElem)))  # eliminate dupicates and sort
            if len(temp_list[-1]) == 1:
                temp_list.pop(-1)

        list_of_listOfCorefSentenceIndexes = temp_list
        #print("Risultato: ", list_of_listOfCorefSentenceIndexes)

        for listElem in list_of_listOfCorefSentenceIndexes:
            for index, i in enumerate(listElem, start=0):
                if index == 0:
                    listElemPopped = listElem.copy()
                listElemPopped.pop(0)
                for j in listElemPopped:
                    graph.addEdge((indexer(docNumber, i), indexer(docNumber, j)))
                    #print('\nEdges: ', graph.getEdges())


        #print("\n--- Graph print: ---\n")
        #print('Nodes: ', graph.getNodes())
        #print('Edges: ', graph.getEdges())

        radix_nodes = graph.getNodes()

        for (node1, node2) in graph.getEdges():
            if node2 in radix_nodes:
                radix_nodes.remove(node2)

        #print("\n--- Radix nodes: ---\n")
        #print(radix_nodes)

        
        #print("\n--- Named entity recognition sentence level: ---\n")
        for i, sentence in enumerate(document, start=0):
            #print('\n ',i,') ')
            #id2entities[indexer(docNumber, i)] = str(sentence.entities)
            id2entities[indexer(docNumber, i)] = list()
            for entity in sentence.entities:
                id2entities[indexer(docNumber, i)].append(str(entity))
                #print(entity, '({})'.format(entity.type))


        #print('\n\n********* Ending document ', docNumber, ' *********\n\n')

    # ENITTY LINKING with the QUESTION
    questionNLP = nlp(question)
    questionEntities = list()
    for entity in questionNLP.entities:
        questionEntities.append(str(entity))
    questionLinked = False
    sentenceIDs = list(id2sentence.keys())
    sentenceIDs.remove('q')
    if(questionEntities != []):
        for j in sentenceIDs:
            if shareEntities(questionEntities, id2entities[j]):
                graph.addEdge(('q', getRadixNode(j)))
                questionLinked = True
    if not questionLinked:
        for j in sentenceIDs:
            if shareWords(question, id2sentence[j]):
                graph.addEdge(('q', getRadixNode(j)))
                questionLinked = True
    # to verify
    if not questionLinked:
        graph.addEdge(('q', indexer(0, 0)))

    # ENITTY LINKING between SENTENCES
    for i in sentenceIDs:
        for j in sentenceIDs:
            if i != j:
                if shareEntities(id2entities[i], id2entities[j]):
                    graph.addEdge((i, getRadixNode(j)))




    #print("\n--- Id 2 entities: ---\n")
    #print(id2entities)

    #print("\n--- Graph: ---\n")
    #print(graph.getEdges())


    #file = open("CoreferenceGraph.pkl", "wb")
    #pickle.dump(graph, file)
    #file.close()
    #file = open("id2sentence.pkl", "wb")
    #pickle.dump(id2sentence, file)
    #file.close()

    return graph, id2sentence

    

