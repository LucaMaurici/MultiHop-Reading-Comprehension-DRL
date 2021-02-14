from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
       "Canada (French: ) is a country in the northern half of North America. Its ten provinces and three territories extend from the Atlantic to the Pacific and northward into the Arctic Ocean, covering , making it the world's second-largest country by total area and the fourth-largest country by land area. Canada's border with the United States is the world's longest land border. The majority of the country has a cold or severely cold winter climate, but southerly areas are warm in summer. Canada is sparsely populated, the majority of its land territory being dominated by forest and tundra and the Rocky Mountains. About four-fifths of the country's population of 36 million people is urbanized and live near the southern border. Its capital is Ottawa, its largest city is Toronto; other major urban areas include Montreal, Vancouver, Calgary, Edmonton, Quebec City, Winnipeg and Hamilton.\
        Origins of the Charlton Press.\nJames (Jim) Charlton (July 26, 1911 - September 20, 2013) It was Jim Charltons experience as an avid numismatist that inspired him to create a catalogue for coin collectors. In 1926, when Jim Charlton was 15, his older brother Harry Charlton gifted him a United States 1863 Indian Head cent.\
        The Indian Head cent, also known as an Indian Head penny, was a one-cent coin ($0.01) produced by the United States Bureau of the Mint from 1859 to 1909. It was designed by James Barton Longacre, the Chief Engraver at the Philadelphia Mint."
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
# primo valore è il numero del doc mentre il secondo è il counter delle parole tra tutti i documenti
print(X)

print(vectorizer.get_feature_names())
#print(X.shape)