# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant les outils de Pytorch)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

import gzip,numpy,torch

import gzip, numpy, torch


if __name__ == '__main__':
	batch_size = 5 # nombre de données lues à chaque fois
	nb_epochs = 10 # nombre de fois que la base de données sera lue
	eta = 0.00001 # taux d'apprentissage
	hidden_layer_size = 350
	# on lit les données
	((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))
	# on crée les lecteurs de données
	train_dataset = torch.utils.data.TensorDataset(data_train,label_train)
	test_dataset = torch.utils.data.TensorDataset(data_test,label_test)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

	# on initialise le modèle et ses poids
	shallow_n = torch.nn.Sequential(
		torch.nn.Linear(data_train.shape[1],hidden_layer_size),
		torch.nn.Sigmoid(),
		torch.nn.Linear(hidden_layer_size,label_train.shape[1])
	)

	# torch.nn.init.uniform_(model_invisible.weight,-0.001,0.001)

	# torch.nn.init.uniform_(model.weight,-math.sqrt((1/hidden_layer_size)),math.sqrt(1/hidden_layer_size))
	# on initiliase l'optimiseur
	loss_func = torch.nn.MSELoss(reduction='sum')
	optim = torch.optim.SGD(shallow_n.parameters(), lr=eta)

	for n in range(nb_epochs):
		# on lit toutes les données d'apprentissage
		for x,t in train_loader:
			# on calcule la sortie du modèle
			y = shallow_n(x)
			# on met à jour les poids
			loss = loss_func(t,y)
			loss.backward()
			optim.step()
			optim.zero_grad()
			
		# test du modèle (on évalue la progression pendant l'apprentissage)
		acc = 0.
		# on lit toutes les donnéees de test
		for x,t in test_loader:
			# on calcule la sortie du modèle
			y = shallow_n(x)
			# on regarde si la sortie est correcte
			acc += torch.argmax(y,1) == torch.argmax(t,1)
		# on affiche le pourcentage de bonnes réponses
		print(acc/data_test.shape[0])