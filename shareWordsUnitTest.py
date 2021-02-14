from pynlp import StanfordCoreNLP
#def createNlp():
annotators = 'tokenize, ssplit, pos, lemma, ner, entitymentions, coref, sentiment, openie'
#annotators = ''
options = {'openie.resolve_coref': True}

nlp = StanfordCoreNLP(annotators=annotators, options=options)

def shareWords(questOrAns, sentence2):
    print("\n✪✪✪ SHARE WORDS: ✪✪✪\n")
    questOrAns = questOrAns.lower()
    for token in nlp(questOrAns)[0]:
        print(token, '->', token.pos)
        print(token, '->', token.lemma)
        if token.pos == 'IN' or token.pos=='CC' or token.pos=='DT' or token.pos=='PRP' or token.pos=='PRP$' or token.pos=='TO' \
            or token.pos == 'WDT' or token.pos == 'WP' or token.pos == 'WP$' or token.pos == 'WRB':
            print('EEE')
            questOrAns = questOrAns.replace(' ' + str(token) + ' ', ' ')
            #print(token.pos)
            print(questOrAns)
        if token.lemma=='be' or token.lemma=='have':
            questOrAns = questOrAns.replace(' ' + str(token) + ' ', ' ')
    print("\n✪✪✪ SENTENCE: ✪✪✪\n")
    print(questOrAns)
    list1 = questOrAns.lower().split()
    list2 = sentence2.lower().split()
    for e1 in list1:
        if e1 in list2:
            return True
    return False

#shareWords('Valerio, from Italy, is going to have dinner and so he is cooking the ragù, a carrot, and it is so dense', 'What is Valerio cooking?')
#shareWords('beautiful', 'What is Valerio cooking?')
shareWords('The University of Wales Trinity Saint David is a collegiate university operating on three main campuses in South West Wales: in Carmarthen, Lampeter, and Swansea. The university also has a fourth campus in London, England.', '')