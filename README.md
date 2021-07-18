# MultiHop Reading Comprehension-DRL

Paper Datasets: https://www.aclweb.org/anthology/Q18-1021.pdf
Wikihop in SQuAD format: https://github.com/alontalmor/MultiQA

FATTO: Controllare se, quando si costruisce lo stato, il numero di azioni venga effettivamente costretto a non essere > 8
minore di 8 già funziona, nel senso che viene aggiunto il è padding in maniera funzionante.

TODO: 	- Droppare dal dataset di train i sample per cui non è chiaro dove sia la risposta
		- Cambiare architettura della rete
		- Usare embeddings preallenati
		- Preprocessing
		- Cambiare entity linking nella costruzione del grafo
		- Capire se si possano fornire i candidates al MnemonicReader
		- Controllare e pensare a contemplare il caso in cui non si riesca a collegare la domanda a nessuna frase
		- Cambiare il DONE nello step dell'environment
		- Spostare la creazione dei link fra documenti consecutivi alla fine di tutto o comunque dopo la coreference resolution

		OBIETTIVO: 
		Arrivare a 33% di accuracy con random walk (tenendo il resto uguale)

		- Stimare media e varianza del grado del grafo
		- Stimare quanto è lungo il cammino ottimo medio fino alla risposta (3,44)
		- Stimare quante volte non si raggiunge in maniera ottima un nodo risposta con più di 10 hop (1,52%)

		Se va tutto bene allora il problema è più probabile che sia nel mnemonic reader "use it as the base reader implementation"
		Altrimenti aggiustare prima il grafo e poi pensare al resto.

		- Provare i candidates con RandomWalk
		- Il mnemonic reader potrebbe avere sia il problema che loro hanno usato quel paper come base implementation e sia il fatto che potrebbero averlo riallenato (forse non capisce le domande nel formato strano di wikihop)

		ALTRO POSSIBILE OBIETTIVO:
		Non usare neanche il mnemonic reader usando come metrica quella in tabella 2, in cui misuriamo quante volte trovaimo una frase con la risposta usando o il grafo con RandomWalk o il grafo con la policy


		RICERCA:
		- Esplorare il grafo dando reward 1 per tutte le frasi che sono ritenute delle possibili contenitrici di risposte. Per una data frase dare reward 1 una sola volta e reward negativa (es: -0.2) per ogni volta che si ripassa due volte sulla stessa frase (forse anche per ogni generica frase). In questo modo cerchiamo di raccogliere tutte le frasi in cui presumiamo possa essere contenuta la risposta e non ci fermiamo alla prima che troviamo (che ptorebbe non contenerla), potremmo addirittura pensare di fermarci (done = True) solo quando siamo passati in tutte le frasi che potrebbero contenere la risposta.
		- Valutare l'idea di allungare gli step consentiti prima di fermarsi (>30)
		- Dare in input alla rete anche gli id delle frasi al fine di estrarre possibili pattern (ad esempio ci sembra che le risposte capitino verso l'inizio dei documenti (in realtà dovremmo controllare quanto sono lunghi in media))
		- Prendere tutti i cammini o n cammini di lunghezza l dalla domanda