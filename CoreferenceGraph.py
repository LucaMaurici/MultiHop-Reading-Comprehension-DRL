def indexer(docNumber, sentenceIndex):
	return 'd'+str(docNumber) + 's'+str(sentenceIndex)

def getRadixNode(node):
	dPos = node.index('s')
	docNumber = node[1:dPos]
	print(indexer(docNumber, 0))
	return indexer(docNumber, 0)

def shareEntities(list1, list2):
	for e1 in list1:
		if e1 in list2:
			return True
	return False

def buildCoreferenceGraph():

	from pynlp import StanfordCoreNLP
	from nltk import tokenize
	import nltk
	nltk.download('punkt')
	import Graph_class as gs
	import pickle

	annotators = 'tokenize, ssplit, pos, lemma, ner, entitymentions, coref, sentiment, openie'
	options = {'openie.resolve_coref': True}

	nlp = StanfordCoreNLP(annotators=annotators, options=options)


	documents = [
	   "Canada (French: ) is a country in the northern half of North America. Its ten provinces and three territories extend from the Atlantic to the Pacific and northward into the Arctic Ocean, covering , making it the world's second-largest country by total area and the fourth-largest country by land area. Canada's border with the United States is the world's longest land border. The majority of the country has a cold or severely cold winter climate, but southerly areas are warm in summer. Canada is sparsely populated, the majority of its land territory being dominated by forest and tundra and the Rocky Mountains. About four-fifths of the country's population of 36 million people is urbanized and live near the southern border. Its capital is Ottawa, its largest city is Toronto; other major urban areas include Montreal, Vancouver, Calgary, Edmonton, Quebec City, Winnipeg and Hamilton.",
	   "Origins of the Charlton Press.\nJames (Jim) Charlton (July 26, 1911 - September 20, 2013) It was Jim Charltons experience as an avid numismatist that inspired him to create a catalogue for coin collectors. In 1926, when Jim Charlton was 15, his older brother Harry Charlton gifted him a United States 1863 Indian Head cent.",
	   "The Indian Head cent, also known as an Indian Head penny, was a one-cent coin ($0.01) produced by the United States Bureau of the Mint from 1859 to 1909. It was designed by James Barton Longacre, the Chief Engraver at the Philadelphia Mint.",
	   "James E. Charlton ( July 26 , 1911 -- September 20 , 2013 ) was a Canadian coin dealer and numismatic publisher . After working as a stationary engineer , he opened a coin store in Toronto . He published his first guidebook , Catalogue of Canadian Coins , Tokens & Fractional Currency , in 1952 . Other titles from Charlton Press followed . Subsequently he sold his store to concentrate on his publications , and later sold his publishing company , Charlton Press , which continues to publish price guides for coins , banknotes and other collectibles . He turned 100 in July 2011 .",
	   "Los Angeles (Spanish for \"The Angels\"), officially the City of Los Angeles and often known by its initials L.A., is the second-most populous city in the United States (after New York City), the most populous city in California and the county seat of Los Angeles County. Situated in Southern California, Los Angeles is known for its mediterranean climate, ethnic diversity, sprawling metropolis, and as a major center of the American entertainment industry. Los Angeles lies in a large coastal basin surrounded on three sides by mountains reaching up to and over .",
	   "A coin is a small, flat, round piece of metal or plastic used primarily as a medium of exchange or legal tender. They are standardized in weight, and produced in large quantities at a mint in order to facilitate trade. They are most often issued by a government.",
	   "Mexico City, officially City of Mexico (, ; abbreviated as \"CDMX\"), is the capital and most populous city of the United Mexican States. As an \"alpha\" global city, Mexico City is one of the most important financial centers in the Americas. It is located in the Valley of Mexico (\"Valle de M\u00e9xico\"), a large valley in the high plateaus at the center of Mexico, at an altitude of . The city consists of sixteen municipalities (previously called boroughs).",
	   "A global city, also called world city or sometimes alpha city or world center, is a city generally considered to be an important node in the global economic system. The concept comes from geography and urban studies, and the idea that globalization can be understood as largely created, facilitated, and enacted in strategic geographic locales according to a hierarchy of importance to the operation of the global system of finance and trade.",
	   "The Greater Toronto Area (GTA) is the most populous metropolitan area in Canada. At the 2011 census, it had a population of 6,054,191, and the census metropolitan area had a population of 5,583,064. The Greater Toronto Area is defined as the central city of Toronto, and the four regional municipalities that surround it: Durham, Halton, Peel, and York. The regional span of the Greater Toronto Area is sometimes combined with the city of Hamilton, Ontario and its surrounding region, to form the Greater Toronto and Hamilton Area. The Greater Toronto Area is the northern part of the Golden Horseshoe.",
	   "Chicago (or ), officially the City of Chicago, is the third-most populous city in the United States, and the fifth-most populous city in North America. With over 2.7\u00a0million residents, it is the most populous city in the state of Illinois and the Midwestern United States, and the county seat of Cook County. The Chicago metropolitan area, often referred to as Chicagoland, has nearly 10\u00a0million people and is the third-largest in the U.S.",
	   "Toronto is the most populous city in Canada, the provincial capital of Ontario, and the centre of the Greater Toronto Area, the most populous metropolitan area in Canada. Growing in population, the 2011 census recorded a population of 2,615,060. As of 2015, the population is now estimated at 2,826,498, making Toronto the fourth-largest city in North America based on the population within its city limits. Toronto trails only Mexico City, New York City, and Los Angeles by this measure, while it is the fifth-largest (behind also Chicago) if ranked by the size of its metropolitan area . An established global city, Toronto is an international centre of business, finance, arts, and culture, and widely recognized as one of the most multicultural and cosmopolitan cities in the world.",
	   "The City of New York, often called New York City or simply New York, is the most populous city in the United States. With an estimated 2015 population of 8,550,405 distributed over a land area of just , New York City is also the most densely populated major city in the United States. Located at the southern tip of the state of New York, the city is the center of the New York metropolitan area, one of the most populous urban agglomerations in the world. A global power city, New York City exerts a significant impact upon commerce, finance, media, art, fashion, research, technology, education, and entertainment, its fast pace defining the term \"New York minute\". Home to the headquarters of the United Nations, New York is an important center for international diplomacy and has been described as the cultural and financial capital of the world.",
	   "Ontario, one of the 13 provinces and territories of Canada, is located in east-central Canada. It is Canada's most populous province by a large margin, accounting for nearly 40 percent of all Canadians, and is the second-largest province in total area. Ontario is fourth-largest in total area when the territories of the Northwest Territories and Nunavut are included. It is home to the nation's capital city, Ottawa, and the nation's most populous city, Toronto."
	   ]

	graph = gs.Graph()

	id2sentence = {}
	id2entities = {}
	#nodes2radix = {}


	for docNumber, text in enumerate(documents, start=0):

		print('\n\n********* STARTING DOCUMENT ', docNumber, ' *********\n\n')

		#text = ("Canada (French: ) is a country in the northern half of North America. Its ten provinces and three territories extend from the Atlantic to the Pacific and northward into the Arctic Ocean, covering , making it the world's second-largest country by total area and the fourth-largest country by land area. Canada's border with the United States is the world's longest land border. The majority of the country has a cold or severely cold winter climate, but southerly areas are warm in summer. Canada is sparsely populated, the majority of its land territory being dominated by forest and tundra and the Rocky Mountains. About four-fifths of the country's population of 36 million people is urbanized and live near the southern border. Its capital is Ottawa, its largest city is Toronto; other major urban areas include Montreal, Vancouver, Calgary, Edmonton, Quebec City, Winnipeg and Hamilton.")
		document = nlp(text)
		#print(document) # prints 'text'

		'''
		print("\n--- SENTENCE SPLITTING: ---\n")
		for index, sentence in enumerate(document):
		    print(index, sentence, sep=' )')
		    id2sentence[indexer(docNumber, index)] = sentence
		    #print("\n---------------------\n")
		    #help(sentence)
		'''
		

		print("\n--- SENTENCE SPLITTING and AddToGraph: ---\n")
		for sentenceIndex, sentence in enumerate(tokenize.sent_tokenize(text), start=0):
			#print("\n---------------------\n")
			#print(sentence)
			#print("\n---------------------\n")
			id2sentence[indexer(docNumber, sentenceIndex)] = sentence
			graph.addNode(indexer(docNumber, sentenceIndex))
			if sentenceIndex != 0:
				graph.addEdge((indexer(docNumber, sentenceIndex-1), indexer(docNumber, sentenceIndex)))


		print("\n--- id2sentence: ---\n")
		print(id2sentence)
		

		print("\n--- Coreference resolution: ---\n")
		for var in document.coref_chains:
			print(var)



		print("\n--- Coreference resolution chain: ---\n")
		list_of_listOfCorefSentenceIndexes = list()
		for i, chain in enumerate(document.coref_chains, start=0):
			#print()
			#print(chain._coref.mention)
			list_of_listOfCorefSentenceIndexes.append(list())
			for mention in chain._coref.mention:
				list_of_listOfCorefSentenceIndexes[i].append(mention.sentenceIndex)


		print()
		print("Prima: ", list_of_listOfCorefSentenceIndexes)
		temp_list = list()
		for listElem in list_of_listOfCorefSentenceIndexes:
			temp_list.append(list(set(listElem)))  # eliminate dupicates and sort
			if len(temp_list[-1]) == 1:
				temp_list.pop(-1)

		list_of_listOfCorefSentenceIndexes = temp_list
		print("Risultato: ", list_of_listOfCorefSentenceIndexes)


		#for listElem in list_of_listOfCorefSentenceIndexes:

		for listElem in list_of_listOfCorefSentenceIndexes:
			for index, i in enumerate(listElem, start=0):
				if index == 0:
					listElemPopped = listElem.copy()
				#print('\nPrima di pop',i,': ', listElemPopped)
				listElemPopped.pop(0)
				#print('\nDopo pop',i,': ', listElemPopped)
				for j in listElemPopped:
					graph.addEdge((indexer(docNumber, i), indexer(docNumber, j)))
					print('\nEdges: ', graph.getEdges())



		print("\n--- Graph print: ---\n")
		print('Nodes: ', graph.getNodes())
		print('Edges: ', graph.getEdges())

		radix_nodes = graph.getNodes()

		for (node1, node2) in graph.getEdges():
			if node2 in radix_nodes:
				radix_nodes.remove(node2)

		print("\n--- Radix nodes: ---\n")
		print(radix_nodes)

		
		print("\n--- Named entity recognition sentence level: ---\n")
		for i, sentence in enumerate(document, start=0):
			#first_sentence = document[0]
			print('\n ',i,') ')
			#id2entities[indexer(docNumber, i)] = str(sentence.entities)
			id2entities[indexer(docNumber, i)] = list()
			for entity in sentence.entities:
				id2entities[indexer(docNumber, i)].append(str(entity))
				print(entity, '({})'.format(entity.type))


		print('\n\n********* Ending document ', docNumber, ' *********\n\n')
		

	for i in id2sentence.keys():
		for j in id2sentence.keys():
			if i != j:
				if shareEntities(id2entities[i], id2entities[j]):
					graph.addEdge((i, getRadixNode(j)))


	print("\n--- Id 2 entities: ---\n")
	print(id2entities)

	print("\n--- Graph: ---\n")
	print(graph.getEdges())


	file = open("CoreferenceGraph.pkl", "wb")
	pickle.dump(graph, file)
	file.close()

	return graph

	

