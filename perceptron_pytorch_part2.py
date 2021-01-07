import gzip, numpy, torch, math
    
if __name__ == '__main__':
	batch_size = 5 # nombre de données lues à chaque fois
	nb_epochs = 10 # nombre de fois que la base de données sera lue
	eta = 0.00001 # taux d'apprentissage
	
	hidden_layer_size = 100
	# on lit les données
	((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))

	# on initialise le modèle et ses poids
	hidden_layer = torch.empty((data_train.shape[1],hidden_layer_size),dtype=torch.float)
	b_hidden_layer = torch.empty((1,hidden_layer_size),dtype=torch.float)

	w = torch.empty((hidden_layer_size,label_train.shape[1]),dtype=torch.float)
	b = torch.empty((1,label_train.shape[1]),dtype=torch.float)

	torch.nn.init.uniform_(w,-0.001,0.001)
	torch.nn.init.uniform_(b,-0.001,0.001)

	torch.nn.init.uniform_(hidden_layer, -math.sqrt((1/hidden_layer_size)),math.sqrt(1/hidden_layer_size))
	torch.nn.init.uniform_(b_hidden_layer, -math.sqrt((1/hidden_layer_size)),math.sqrt(1/hidden_layer_size))

	nb_data_train = data_train.shape[0]
	nb_data_test = data_test.shape[0]

	indices = numpy.arange(nb_data_train,step=batch_size)
	for n in range(nb_epochs):
		# on mélange les (indices des) données
		numpy.random.shuffle(indices)
		# on lit toutes les données d'apprentissage
		for i in indices:
			# on récupère les entrées
			x = data_train[i:i+batch_size]

			# on calcule la sortie de la première couche
			expo = -(torch.mm(x, hidden_layer) + b_hidden_layer)
			hidden_y = 1/(1 + torch.exp(expo)) 

			#on calcule la sortie final
			y = torch.mm(hidden_y,w) + b

			# on calcule la sortie du modèle
			t = label_train[i:i+batch_size]
			
			grad = (t - y)
			hidden_grad = hidden_y *(1-hidden_y) * torch.mm(grad,w.T)


			#on met à jour les poids
			hidden_layer += eta * torch.mm(x.T,hidden_grad)
			b_hidden_layer += eta * hidden_grad.sum(axis=0)

			w += eta * torch.mm(hidden_y.T,grad)
			b += eta * grad.sum(axis=0)
		# test du modèle (on évalue la progression pendant l'apprentissage)
		acc = 0.
		# on lit toutes les donnéees de test
		for i in range(nb_data_test):
			# on récupère l'entrée
			x = data_test[i:i+1]
			# on calcule la sortie du modèle
			expo = -(torch.mm(x, hidden_layer) + b_hidden_layer)
			hidden_y = 1/(1 + torch.exp(expo)) 

			#on calcule la sortie final
			y = torch.mm(hidden_y,w) + b

			# on regarde le vrai label
			t = label_test[i:i+1]
			# on regarde si la sortie est correcte
			acc += torch.argmax(y,1) == torch.argmax(t,1)
		# on affiche le pourcentage de bonnes réponses
		print(acc/nb_data_test)
