# Artificial Neural Network
# Installing Theano    
#pip install theano      
# Installing Tensorflow 
#pip install tensorflow     
# Installing Keras     
#pip install Keras

#Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
raw_dataset=pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#########################################
# Encoding cathegorical data
X = dataset.iloc[:, 3:13].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:, 1:]        #http://www.algosome.com/articles/dummy-variable-trap-regression.html
                    #Dummy variable trap ! be aware of !

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
##END OF DATA PREPROCESSING

# Part 2 - No lets make an ANN ! 
#Iporting the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(output_dim=6,init='uniform', activation='relu', input_dim=11))
classifier.add(Dropout(p=0.1))
# adding the second hidden layer
classifier.add(Dense(output_dim=6,init='uniform', activation='relu'))
classifier.add(Dropout(p=0.1))
# adding the output layer
classifier.add(Dense(output_dim=1,init='uniform', activation='sigmoid')) #depended soft max function if more than two cathegorical data
#Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10 ,nb_epoch=100)
#Part 3 - Makign the predictions and evaluating the model



# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#predicting single new observation
"""
GEO: France
Credit score: 600
gender: Male
Age: 40
Tenure:3
Balance:60000
Number of products:2
Has credit card: yes
is active member: yes
estimated salary: 50000
"""
new_prediction=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000, 2, 1, 1, 50000]])))
new_prediction=(new_prediction>0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)     

# Part 4 Evaluating, oproving and tunign the ANN
#Evaluating the ANN


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout #randomly disable neurons
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6,init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform', activation='sigmoid')) 
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )
    return classifier
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)
accuracies = cross_val_score(estimator=classifier,X=X_train, y=y_train, cv=10, n_jobs=1)
mean=accuracies.mean()
variance=accuracies.std()
#improving the ANN
#Dropout regularization to reduce overfitting if needed

#tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout #randomly disable neurons
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim=6,init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform', activation='sigmoid')) 
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'] )
    return classifier
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size' : [25, 32],
              'nb_epoch':[100, 500],
              'optimizer':['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring = 'accuracy',
                           cv=10)
grid_search=grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

