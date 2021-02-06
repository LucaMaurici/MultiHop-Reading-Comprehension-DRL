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

def shareWords(sentence1, sentence2):

    list1 = sentence1.lower().split()
    list2 = sentence2.lower().split()
    for e1 in list1:
        if e1 in list2:
            return True
    return False

def buildCoreferenceGraph(question, documents):

    from pynlp import StanfordCoreNLP
    from nltk import tokenize
    import nltk
    nltk.download('punkt')
    import Graph_class as gs
    import pickle

    annotators = 'tokenize, ssplit, pos, lemma, ner, entitymentions, coref, sentiment, openie'
    options = {'openie.resolve_coref': True}

    nlp = StanfordCoreNLP(annotators=annotators, options=options)


    '''
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
    '''

    '''
    documents = [
        "The Cascade Range or Cascades is a major mountain range of western North America, extending from southern British Columbia through Washington and Oregon to Northern California. It includes both non-volcanic mountains, such as the North Cascades, and the notable volcanoes known as the High Cascades. The small part of the range in British Columbia is referred to as the Canadian Cascades or, locally, as the Cascade Mountains. The latter term is also sometimes used by Washington residents to refer to the Washington section of the Cascades in addition to North Cascades, the more usual U.S. term, as in North Cascades National Park. The highest peak in the range is Mount Rainier in Washington at .",
        "The Okanogan National Forest is a U.S. National Forest located in Okanogan County in north-central Washington, United States.",
        "The United States Forest Service (USFS) is an agency of the U.S. Department of Agriculture that administers the nation's 154 national forests and 20 national grasslands, which encompass . Major divisions of the agency include the National Forest System, State and Private Forestry, Business Operations, and the Research and Development branch. Managing approximately 25% of federal lands, it is the only major national land agency that is outside the U.S. Department of the Interior.",
        "Mount Rainier National Park is a United States National Park located in southeast Pierce County and northeast Lewis County in Washington state. It was established on March 2, 1899 as the fifth national park in the United States. The park encompasses including all of Mount Rainier, a stratovolcano. The mountain rises abruptly from the surrounding land with elevations in the park ranging from 1,600 feet to over 14,000 feet (490 - 4,300\u00a0m). The highest point in the Cascade Range, around it are valleys, waterfalls, subalpine meadows, old-growth forest and more than 25 glaciers. The volcano is often shrouded in clouds that dump enormous amounts of rain and snow on the peak every year.",
        "Snoqualmie National Forest is a United States National Forest in the State of Washington. It was established on 1 July 1908, when an area of 961,120 acres (3,889.52 km\u00b2) was split from the existing Washington National Forest. Its size was increased on 13 October 1933, when a part of Rainier National Forest was added. In 1974 Snoqualmie was administratively combined with Mount Baker National Forest to make Mount Baker-Snoqualmie National Forest. In descending order of land area, Snoqualmie National Forest lies in parts of King, Snohomish, Pierce, and Kittitas counties. There are local ranger district offices in North Bend and Skykomish. Its main base is in Everett, Washington. As of 30 September 2007, it had an area of 1,258,167 acres (5,091.62 km\u00b2), representing about 49 percent of the combined forest's total acreage.",
        "Chelan National Forest was established in Washington by the U.S. Forest Service on July 1, 1908 with from a portion of Washington National Forest. On July 1, 1921 it absorbed the first Okanogan National Forest, but on March 23, 1955 the name was changed back to Okanogan.",
        "Mount Baker (Lummi: '; or '), also known as Koma Kulshan or simply Kulshan, is an active glaciated andesitic stratovolcano in the Cascade Volcanic Arc and the North Cascades of Washington in the United States. Mount Baker has the second-most thermally active crater in the Cascade Range after Mount Saint Helens. About due east of the city of Bellingham, Whatcom County, Mount Baker is the youngest volcano in the Mount Baker volcanic field. While volcanism has persisted here for some 1.5 million years, the current glaciated cone is likely no more than 140,000 years old, and possibly no older than 8090,000 years. Older volcanic edifices have mostly eroded away due to glaciation.",
        "Mount Baker National Forest was established in Washington on January 21, 1924 when its name was changed from Washington National Forest. In 1974 it was administratively combined with Snoqualmie National Forest to make Mount Baker-Snoqualmie National Forest. In descending order of land area, Mount Baker National Forest is located in parts of Snohomish, Whatcom, and Skagit counties. As of 30 September 2007, it had an area of 1,301,787 acres (5,268.1\u00a0km\u00b2), representing about 51 percent of the combined forest's area. There are local ranger district offices located in Darrington and Sedro-Woolley.",
        "The Public Land Survey System (PLSS) is the surveying method developed and used in the United States to plat, or divide, real property for sale and settling. Also known as the Rectangular Survey System, it was created by the Land Ordinance of 1785 to survey land ceded to the United States by the Treaty of Paris in 1783, following the end of the American Revolution. Beginning with the Seven Ranges, in present-day Ohio, the PLSS has been used as the primary survey method in the United States. Following the passage of the Northwest Ordinance, in 1787, the Surveyor General of the Northwest Territory platted lands in the Northwest Territory. The Surveyor General was later merged with the General Land Office, which later became a part of the U.S Bureau of Land Management, or BLM. Today, the BLM controls the survey, sale, and settling of the new lands, and manages the State Plane Coordinate System.",
        "The Land Ordinance of 1785 was adopted by the United States Congress of the Confederation on May 20, 1785. It set up a standardized system whereby settlers could purchase title to farmland in the undeveloped west. Congress at the time did not have the power to raise revenue by direct taxation, so land sales provided an important revenue stream. The Ordinance set up a survey system that eventually covered over three-fourths of the area of the continental United States.",
        "The Mount Baker-Snoqualmie National Forest in Washington is a National Forest extending more than along the western slopes of the Cascade Range from the CanadaUS border to the northern boundary of Mount Rainier National Park. Administered by the United States Forest Service, the forest is headquartered in Everett.",
        "The Department of the Treasury is an executive department and the treasury of the United States federal government. It was established by an Act of Congress in 1789 to manage government revenue. The Department is administered by the Secretary of the Treasury, who is a member of the Cabinet. Jacob J. Lew is the current Secretary of the Treasury; he was sworn in on February 28, 2013.",
        "The General Land Office (GLO) was an independent agency of the United States government responsible for public domain lands in the United States. It was created in 1812 to take over functions previously conducted by the United States Department of the Treasury. Starting with the passage of The Land Ordinance of 1785, which created the Public Land Survey System, the Treasury Department had already overseen the survey of the \"Northwest Territory\" including what is now the State of Ohio.",
        "Washington National Forest was established by the General Land Office as the Washington Forest Reserve in Washington on February 22 , 1897 with 3,594,240 acres ( 14,545.4 km2 ) . After the transfer of federal forests to the U.S. Forest Service in 1905 , it became a National Forest on March 4 , 1907 . On July 1 , 1908 , Chelan National Forest was established with a portion of Washington . On January 21 , 1924 Washington was renamed Mount Baker National Forest . The lands presently exist as Mount Baker - Snoqualmie National Forest ."
    ]
    '''

    '''
    question = 'publication_date blueprint '
    documents = [
        "McKinley Morganfield (April 4, 1913  April 30, 1983), better known as Muddy Waters, was an American blues musician who is often cited as the \"father of modern Chicago blues\".",
        "The guitar is a musical instrument classified as a string instrument with anywhere from four to 18 strings, usually having six. The sound is projected either acoustically, using a hollow wooden or plastic and wood box (for an acoustic guitar), or through electrical amplifier and a speaker (for an electric guitar). It is typically played by strumming or plucking the strings with the fingers, thumb and/or fingernails of the right hand or with a pick while fretting (or pressing against the frets) the strings with the fingers of the left hand. The guitar is a type of chordophone, traditionally constructed from wood and strung with either gut, nylon or steel strings and distinguished from other chordophones by its construction and tuning. The modern guitar was preceded by the gittern, the vihuela, the four-course Renaissance guitar, and the five-course baroque guitar, all of which contributed to the development of the modern six-string instrument.",
        "Blueprint is the third album by Irish guitarist Rory Gallagher , released as a vinyl record in 1973 . With his first band Taste and with his solo band up to this point Gallagher was one of the first guitarists to lead a power trio lineup . With Blueprint Gallagher included a keyboardist for the first time .",
        "William Rory Gallagher (; 2 March 1948\u00a0 14 June 1995) was an Irish blues and rock multi-instrumentalist, songwriter, and bandleader. Born in Ballyshannon, County Donegal, and brought up in Cork, Gallagher recorded solo albums throughout the 1970s and 1980s, after forming the band Taste during the late 1960s. He was a talented guitarist known for his charismatic performances and dedication to his craft. Gallagher's albums have sold in excess of 30 million copies worldwide. Gallagher received a liver transplant in 1995, but died of complications later that year in London, UK at the age of 47.",
        "The Hammond organ is an electric organ, invented by Laurens Hammond and John M. Hanert and first manufactured in 1935. Various models have been produced, most of which use sliding drawbars to create a variety of sounds. Until 1975, Hammond organs generated sound by creating an electric current from rotating a metal tonewheel near an electromagnetic pickup, and then strengthening the signal with an amplifier so that it can drive a speaker cabinet. Around two million Hammond organs have been manufactured. The organ is commonly used with, and associated with, the Leslie speaker.",
        "County Donegal (pronounced or ) is a county of Ireland. It is part of the Border Region of the Republic of Ireland and is in the province of Ulster. It is named after the town of Donegal in the south of the county. Donegal County Council is the local council for the county and Lifford serves as the county town. The population of the county is 158,755 according to the 2016 census. It has also been known as (County) Tyrconnell (\"\"), after the historic territory of the same name.",
        "A power trio is a rock and roll band format having a lineup of guitar, bass and drums, leaving out the second guitar or keyboard that are used in other rock music quartets and quintets to fill out the sound with chords. While one or more band members typically sing, power trios emphasize instrumental performance and overall impact over vocals and lyrics.\nHistory.\nThe rise of the power trio in the 1960s was made possible in part by developments in amplifier technology that greatly enhanced the volume of the electric guitar and bass. Particularly, the popularization of the electric bass guitar defined the bottom end and filled in the gaps. Since the amplified bass could also now be louder, the rest of the band could also play at higher volumes, without fear of being unable to hear the bass. This allowed a three-person band to have the same sonic impact as a large band but left far more room for improvisation and creativity, unencumbered by the need for detailed arrangements. As with the organ trio, a 1960s-era soul jazz group centered on the amplified Hammond organ, a three-piece group could fill a large bar or club with a big sound for a much lower price than a large rock and roll band. A power trio, at least in its blues rock incarnation, is also generally held to have developed out of Chicago-style blues bands such as Muddy Waters' trio."
    ]
    '''

    temp = list()
    for document in documents:
        document = document.replace('%', ' percent')
        #document = document.replace('+', '<plus>')
        temp.append(document)
    documents = temp
    print(documents)

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

    # ENITTY LINKING with the QUESTION
    questionNLP = nlp(question)
    questionEntities = list()
    for entity in questionNLP.entities:
        questionEntities.append(str(entity))
    print(questionEntities)
    if(questionEntities != []):
        for j in id2sentence.keys():
            if shareEntities(questionEntities, id2entities[j]):
                graph.addEdge(('q', getRadixNode(j)))
    else:
        for j in id2sentence.keys():
            if shareWords(question, id2sentence[j]):
                graph.addEdge(('q', getRadixNode(j)))

    # ENITTY LINKING between SENTENCES
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
    file = open("id2sentence.pkl", "wb")
    pickle.dump(id2sentence, file)
    file.close()

    return graph, id2sentence

    

