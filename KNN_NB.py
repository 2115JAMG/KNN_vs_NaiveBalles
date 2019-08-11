"""
Josué Alexis M.G.
09-08-19
KNN and NB
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

"se carga la base de datos en una variable"
"it´s loaded the database in a variable"
df = pd.read_csv('C:\\automobile.csv')
print(df.head(3))

"variables para entrenamiento"
"variables to training"
x = df[['longitud','ancho','peso']]
y = df['Traccion']

"30% test      70% training"
"división de mi base de datos para entrenamiento y test"
"split of my database to test and training"
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

"escala los valores"
"scales the values"
scaler= MinMaxScaler()

x_trainS = scaler.fit_transform(x_train)
x_testS = scaler.transform(x_test)

"valor de vecinos KNN"
"value of neighbors KNN"
n_neighbors =7 

knn = KNeighborsClassifier(n_neighbors)
knn.fit(x_trainS, y_train)

y_predKNN = knn.predict(x_testS)

print(accuracy_score(y_test, y_predKNN))

print(confusion_matrix(y_test, y_predKNN))
print(classification_report(y_test, y_predKNN))

predecir = scaler.fit_transform([[160.3, 65, 53.5]])
print(knn.predict(predecir))

nb =GaussianNB()
nb.fit(x_train, y_train)

y_predNB = nb.predict(x_test)
"imprimir en terminal"
"print to terminal"

"presición de mi sistema"
"acuracy of my system"
print(confusion_matrix(y_test, y_predNB))
print(classification_report(y_test, y_predNB))

xS= scaler.fit_transform(x)

puntajesKNN = cross_val_score(knn, xS, y, cv=5)
puntajesNB = cross_val_score(nb, xS, y, cv=5)
"imprimir en terminal"
"print to terminal"

print( "Accuracy: %0.2f (+/-%0.2f)" %(puntajesKNN.mean(), puntajesKNN.std() *2))
print( "Accuracy: %0.2f (+/-%0.2f)" %(puntajesNB.mean(), puntajesNB.std() *2))
""" Es importante checar la variabilidad de mi aacuracy"""