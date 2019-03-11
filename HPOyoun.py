#First we import the things that you do not want to do by ourself :-)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import *
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import numpy as np
import csv
import random
from pprint import pprint
#IF WE WANT SOMETHING DETERMINISTIC
#np.random.seed(5);
def build_model(optimizer, learning_rate, activation, dropout_rate,num_unit):
    keras.backend.clear_session()
    model = Sequential()
    model.add(Dense(num_unit, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_unit, activation=activation))
    model.add(Dropout(dropout_rate)) 
    model.add(Dense(int(num_unit/2), activation=activation))
    model.add(Dense(1, activation=activation))
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer(lr=learning_rate),
                  metrics=['accuracy'])
    return model


batch_size = [20, 10, 30, 50, 100]

epochs = [1, 10 ,20, 50]

learning_rate = [0.1, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002 ,0.001, 0.02]

dropout_rate = [ 0.2, 0.3, 0.8]

num_unit = [10, 6, 36, 50, 100]

activation = [ 'softplus', 'tanh', 'relu', 'sigmoid']

optimizer = [SGD][:1]
#loading external dataset (CSV file)


dataset = np.loadtxt('dataAND.csv', delimiter=",")

parameters = dict(batch_size = batch_size,
                  epochs = epochs,
                  dropout_rate = dropout_rate,
                  num_unit = num_unit,
                  learning_rate = learning_rate,
                  activation = activation,
                  optimizer = optimizer)

model = KerasClassifier(build_fn=build_model, verbose=0)

models = GridSearchCV(estimator = model, param_grid=parameters, n_jobs=1, cv=10, error_score='raise', refit='True')

#mixing data randomly
random.shuffle(dataset)


x_train= np.array(dataset[:,0:10])
y_train=dataset[:,10:11] #the class is the last element in a row



best_model = models.fit(x_train, y_train)
print('Best model :')
pprint(best_model.best_params_)

pprint(models.cv_results_['params'][models.best_index_])

optimizer = best_model.best_params_['optimizer']
learning_rate = best_model.best_params_['learning_rate']
activation = best_model.best_params_['activation']
dropout_rate = best_model.best_params_['dropout_rate']
num_unit = best_model.best_params_['num_unit']



random.shuffle(dataset)
Folds = {'X':{}, 'Y':{}}
low_bound=0
high_bound=100
for i in range(10):
    Folds['X'][i]=np.array(dataset[low_bound:high_bound,0:10])
    Folds['Y'][i]=np.array(dataset[low_bound:high_bound,10:11])
    low_bound+=100
    high_bound+=100

myFavouriteModel= build_model(optimizer, learning_rate, activation, dropout_rate,num_unit)
myFavouriteModel
ModelesDic={'Model':{},'MeanMSE': 0 , 'MSE':{} }
for m in range(10):
        ModelesDic['Model'][m]=build_model(optimizer, learning_rate, activation, dropout_rate,num_unit)
        for u in Folds['X']:
            if u==i:
                x_test=Folds['X'][u]
                y_test=Folds['Y'][u]
            else:
                if 'x_train' in locals():
                    x_train=np.concatenate((x_train,Folds['X'][u]), axis=0)
                    y_train=np.concatenate((y_train,Folds['Y'][u]), axis=0)
                else:
                    x_train=np.array(Folds['X'][u])
                    y_train=np.array(Folds['Y'][u])
        ModelesDic['Model'][m].fit(x_train, y_train,epochs= best_model.best_params_['epochs'],batch_size=best_model.best_params_['batch_size'])
        score = ModelesDic['Model'][m].evaluate(x_test, y_test, batch_size=best_model.best_params_['batch_size'])
        print('\nFinal accuracy on test set:'+str(score[1]))
        print('\nFinal MSE on test set:'+str(score[0]))
        ModelesDic['MSE'][m]=score[0]
        ModelesDic['MeanMSE']+=score[0]
ModelesDic['MeanMSE']=ModelesDic['MeanMSE']/10
pprint(ModelesDic)

pprint(models)