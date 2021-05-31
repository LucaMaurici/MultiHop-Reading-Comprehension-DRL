# MultiHop Reading Comprehension-DRL

FATTO: Controllare se, quando si costruisce lo stato, il numero di azioni venga effettivamente costretto a non essere > 8
minore di 8 già funziona, nel senso che viene aggiunto il è padding in maniera funzionante.
 
TODO: 	- Droppare dal dataset di train i sample per cui non è chiaro dove sia la risposta
		- Cambiare architettura della rete
		- Usare embeddings preallenati
		- Preprocessing
		- Cambiare entity linking nella costruzione del grafo
		- Capire se si possano fornire i candidates al MnemonicReader
		- Controllare e pensare a contemplare il caso in cui non si riesca a collegare la domanda a nessuna frase