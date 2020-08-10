#pylint:disable=E1101

from sklearn import datasets
iris = datasets.load_iris()

#Observations

print(dir(iris))

target = iris.target

for i in [0,1,2]:
    print("classe : %s, nb exemplaires: %s" % (i, len(target[ target == i]) ) )

data = iris.data

#Representation

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax1 = plt.subplot(1,2,1)

clist = ['violet', 'yellow', 'blue']
colors = [clist[c] for c in iris.target]

ax1.scatter(data[:, 0], data[:, 1], c=colors)
plt.xlabel('Longueur du sepal (cm)')
plt.ylabel('Largueur du sepal (cm)')

ax2 = plt.subplot(1,2,2)

ax2.scatter(data[:, 2], data[:, 3], color=colors)

plt.xlabel('Longueur du petal (cm)')
plt.ylabel('Largueur du petal (cm)')

# Légende
for ind, s in enumerate(iris.target_names):
    # on dessine de faux points, car la légende n'affiche que les points ayant un label
    plt.scatter([], [], label=s, color=clist[ind])

plt.legend(scatterpoints=1, frameon=False, labelspacing=1
           , bbox_to_anchor=(1.8, .5) , loc="center right", title='Espèces')
plt.plot()


import seaborn as sns
import pandas as pd
sns.set()
df = pd.DataFrame(data, columns=iris['feature_names'] )
df['target'] = target
df['label'] = df.apply(lambda x: iris['target_names'][int(x.target)], axis=1)
df.head()

sns.pairplot(df, hue='label', vars=iris['feature_names'], height=2);

#Prediction

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

clf.fit(data, target) # On aurait aussi pu utiliser le dataframe df

result = clf.predict(data)

result - target # Qualité test

#Vérifions la réussite

errors = sum(result != target) # 6 erreurs sur 150 mesures
print("Nb erreurs:", errors)
print( "Pourcentage de prédiction juste:", (150-errors)*100/150)   # 96 % de réussite

#Ou

from sklearn.metrics import accuracy_score
accuracy_score(result, target) # 96% de réussite

#Ou

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(target, result)
#sns.heatmap(conf, square=True, annot=True, cbar=False
#            , xticklabels=list(iris.target_names)
#            , yticklabels=list(iris.target_names))
#plt.xlabel('valeurs prédites')
#plt.ylabel('valeurs réelles')
# ou faire plt.matshow(conf, cmap='rainbow');


#Tester proprement avec deux jeux de données
from sklearn.model_selection import train_test_split # version 0.18.1
data_test = train_test_split(data, target
                                 , random_state=0
                                 , train_size=0.5)
data_train, data_test, target_train, target_test = data_test
clf = GaussianNB()
clf.fit(data_train, target_train)
result = clf.predict(data_test)
# Score
accuracy_score(result, target_test) #0.94666, c'est très bien !

# Matrice de confusion
conf = confusion_matrix(target_test, result)
sns.heatmap(conf, square=True, annot=True, cbar=False
            , xticklabels=list(iris.target_names)
            , yticklabels=list(iris.target_names))
plt.xlabel('valeurs prédites')
plt.ylabel('valeurs réelles')

# On ne conserve que les longueurs/largeurs des sépales
data = iris.data[:, :2]
target = iris.target
data[:5]

# On réapprend
clf = GaussianNB()
clf.fit(data, target)
h = .15
# Nous recherchons les valeurs min/max de longueurs/largeurs des sépales
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

x = np.arange(x_min, x_max, h)
y = np.arange(y_min, y_max, h)

xx, yy = np.meshgrid(x,y )

# http://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html
# applatit les données du tableau
data_samples = list(zip(xx.ravel(), yy.ravel()) )

# La fonction ravel applatit un tableau à n dimensions en 1 tableau d'une dimension:

a = [ [10, 20],
      [ 1,  2] ]
np.array(a).ravel()

# La fonction zip génère quant-à-elle une liste de n-uplets constituée des éléments du même rang de chaque liste reçue en paramètre:

list(zip([10,20,30], [1,2,3]))

Z = clf.predict(data_samples)
#Z = Z.reshape(xx.shape)
plt.figure(1)
#plt.pcolormesh(xx, yy, Z) # Affiche les déductions en couleurs pour les couples x,y

# Plot also the training points
#plt.scatter(data[:, 0], data[:, 1], c=target)
colors = ['violet', 'yellow', 'red']
C = [colors[x] for x in Z]

plt.scatter(xx.ravel(), yy.ravel(), c=C)
plt.xlim(xx.min() - .1, xx.max() + .1)
plt.ylim(yy.min() - .1, yy.max() + .1)
plt.xlabel('Longueur du sepal (cm)')
plt.ylabel('Largueur du sepal (cm)')

# On obtient la clé de détermination


# Avec pcolormesh
plt.figure(1)
plt.pcolormesh(xx, yy, Z.reshape(xx.shape)) # Affiche les déductions en couleurs pour les couples x,y
# Plot also the training points
colors = ['violet', 'yellow', 'red']
C = [colors[x] for x in target]
plt.scatter(data[:, 0], data[:, 1], c=C)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Longueur du sepal (cm)')
plt.ylabel('Largueur du sepal (cm)')


# # # # # # # # # # # # # # # # # # # # 
# # Un monde sans target # # # # # # # 
# # # # # # # # # # # # # # # # # # # # 

# Clustering

from sklearn.decomposition import PCA
# Définition de l'hyperparamètre du nombre de composantes voulues
model = PCA(n_components=2)
# Alimentation du modèle
model.fit(iris.data)
# Transformation avec ses propres données
reduc = model.transform(iris.data )

from sklearn.mixture import GaussianMixture
# Création du modèle avec 3 groupes de données
model = GaussianMixture (n_components=3, covariance_type='full')
# Apprentissage, il n'y en a pas vraiment
model.fit(df[['PCA1', 'PCA2']])
# Prédiction
groups = model.predict(df[['PCA1', 'PCA2']])

df['group'] = groups
sns.lmplot("PCA1", "PCA2", data=df, hue='label',
           col='group', fit_reg=False)

