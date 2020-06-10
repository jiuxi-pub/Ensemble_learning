from pathlib import Path, PureWindowsPath
import itertools
import pandas as pd
import numpy as np
np.random.seed(5555) # for reproducibility
from random import sample
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax
from scipy.sparse import *
from scipy import *
from timeit import default_timer as timer
from sklearn import random_projection
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
# from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# keras use Tensorflow core
from keras.models import Sequential
from keras.layers import Dense
##########################################

# num_splits = 15
rs_val = 12
def bagging(X_train, y_train, portion=0.8, shuffle=True):
    ix = [i for i in range(len(X_train))]
    samples = round(portion * len(X_train))
    train_ix = resample(ix, replace=True, n_samples=samples, random_state=rs_val)
    if(shuffle == False):
        train_ix.sort()
    else: train_ix = train_ix
    test_ix = [x for x in ix if x not in train_ix]
    trainX, trainy = X_train.iloc[train_ix], y_train.iloc[train_ix]
    validX, validy = X_train.iloc[test_ix], y_train.iloc[test_ix]
    # return types are dataframes
    return trainX.to_numpy(), trainy, validX.to_numpy(), validy

def bagging_ndarray(X_train, y_train, portion =0.8, shuffle=True):
    ix = [i for i in range(len(X_train))]
    samples = round(portion* len(X_train))
    train_ix = resample(ix, replace=True, n_samples=samples, random_state=rs_val)
    if(shuffle == False):
        train_ix.sort()
    else: train_ix = train_ix
    test_ix = [x for x in ix if x not in train_ix]
    trainX, trainy = X_train[train_ix], y_train.iloc[train_ix]
    validX, validy = X_train[test_ix], y_train.iloc[test_ix]
    # return types are dataframes
    return trainX, trainy, validX, validy

def ensemble_scoring_plot(members, X_test_thin_sets, y_test_thin, num_splits= 15):
    single_scores, ensemble_scores = [], []
    ax0 = plt.subplot(111)
    for i in range(1, num_splits+1):
        ensemble_score = evaluate_n_members(members, i, X_test_thin_sets, y_test_thin)
        single_score = members[i-1].score(X_test_thin_sets[i-1], y_test_thin)
        print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
        ensemble_scores.append(ensemble_score)
        single_scores.append(single_score)
    # plot score vs number of ensemble members
    print('Accuracy %.3f (%.3f)' % (mean(single_scores), std(single_scores)))
    x_axis = [i for i in range(1, num_splits+1)]
    ax0.plot(x_axis, single_scores, marker='o', linestyle='None')
    ax0.plot(x_axis, ensemble_scores, marker='o')
    plt.show()
    return None

def ensemble_scoring_plot_mlp(members, X_test_thin_sets, y_test_thin, n_splits= 15):
    single_scores, ensemble_scores = [], []
    ax0 = plt.subplot(111)
    for i in range(1, n_splits+1):
        ensemble_score = evaluate_n_members(members, i, X_test_thin_sets, y_test_thin)
        _, single_score = members[i-1].evaluate(X_test_thin_sets[i-1], y_test_thin)
        print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
        ensemble_scores.append(ensemble_score)
        single_scores.append(single_score)
    # plot score vs number of ensemble members
    print('Accuracy %.3f (%.3f)' % (mean(single_scores), std(single_scores)))
    x_axis = [i for i in range(1, n_splits+1)]
    ax0.plot(x_axis, single_scores, marker='o', linestyle='None')
    ax0.plot(x_axis, ensemble_scores, marker='o')
    plt.show()
    return None

def ensemble_scoring(members, X_test_thin_sets, y_test_thin, n_splits=15):
    single_scores, ensemble_scores = [], []
    for i in range(1, n_splits+1):
        ensemble_score = evaluate_n_members(members, i, X_test_thin_sets, y_test_thin)
        single_score = members[i-1].score(X_test_thin_sets[i-1], y_test_thin)
        ensemble_scores.append(ensemble_score)
        single_scores.append(single_score)
    return single_scores, ensemble_scores

def ensemble_scoring_mlp(members, X_test_thin_sets, y_test_thin, n_splits=15):
    single_scores, ensemble_scores = [], []
    for i in range(1, n_splits+1):
        ensemble_score = evaluate_n_members(members, i, X_test_thin_sets, y_test_thin)
        _, single_score = members[i-1].evaluate(X_test_thin_sets[i-1], y_test_thin)
        ensemble_scores.append(ensemble_score)
        single_scores.append(single_score)
    return single_scores, ensemble_scores

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
    # select a subset of members
    subset = members[:n_members]
    # make prediction
    yhat = ensemble_predictions(subset, testX)
    # calculate accuracy
    return accuracy_score(testy, yhat)

# make an ensemble prediction for multi-class classification
def ensemble_predictions(subset, testX):
    # make predictions
    yhats = [model.predict(testX[subset.index(model)]) for model in subset]
    yhats = array(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    # majority voting
    result = np.empty(0)
    for i in summed:
        if (i>len(subset)/2):
            result = np.append(result, 1)
        else:
            result = np.append(result, 0)
    return result

def train_time_avg(model, trainX, trainy, n_avg = 10):
    start = timer()
    for i in range(n_avg):
        # fit model
        model.fit(trainX, trainy)
    end = timer()
    t = end - start
    t_average = t/n_avg
    print ('Average time taken: {}'.format(t_average))
    return t_average

def evaluate_model_sgd(trainX, trainy, testX, testy, avg_time = False, n_avg = 10, early_stopping=True, eta0= 0.2, learning_rate='adaptive'):
    # define model
    model = SGDClassifier(alpha= 0.0001, loss= 'log', max_iter= 1000, validation_fraction= 0.1, random_state=rs_val, early_stopping=early_stopping, eta0=eta0, learning_rate=learning_rate, n_iter_no_change= 6, tol=1e-4, verbose=False)
    if (avg_time):
        train_time_avg(model, trainX, trainy, n_avg = n_avg)
    else:
        # fit model
        model.fit(trainX, trainy)
    # evaluate the model
    test_acc = model.score(testX, testy)
    return model, test_acc

## requires tuning
def evaluate_model_sgd_bs1(trainX, trainy, testX, testy, avg_time = False, n_avg = 10, num_epochs= 1, num_iters = 1000, alpha=0.0001, eta0=0.0001, lr= 'optimal', penalty='l1',verbose = False):
    # define model
    # model parameters get automatically reset for each assignment, this is verified.
    model = SGDClassifier(alpha= alpha, eta0= eta0, loss= 'log', max_iter= 1000, learning_rate= lr, validation_fraction= 0.1, random_state=rs_val, early_stopping=False, n_iter_no_change= 6, tol=1e-4, verbose=verbose, warm_start=True, shuffle=True, penalty=penalty)
    if (avg_time):
        train_time_avg(model, trainX, trainy, n_avg = n_avg)
    else:
        # fit model
        t_total = 0
        for _ in range(num_epochs):
            X_new, y_new = shuffle(trainX, trainy)
            for index in range(len(X_new)):
                if (index == num_iters):
                    break
                X = X_new[index].reshape(1,-1)
                y = y_new.iloc[index].reshape(1,)
                start = timer()
                model.partial_fit(X, y, np.unique(y_new))
                end  = timer()
                t = end - start
                t_total += t
            print ("Time taken: {}".format(t_total))
            SGDfat_acc = model.score(testX, testy)
            # print (model.intercept_)
            print ('SGDFat accuracy: {}'.format(SGDfat_acc))
    # evaluate the model
    test_acc = model.score(testX, testy)
    return model, test_acc

def evaluate_model_mlp(trainX, trainy, testX, testy, input_dim = 1000, batch_size = 100, num_epochs=50, verbose = 0, shuffle=True):
    # define model
    model = Sequential()
    model.add(Dense(50, input_dim=input_dim, activation='relu'))
    model.add(Dense(50, activation= 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    start = timer()
    model.fit(trainX, trainy, batch_size=batch_size, epochs=num_epochs, verbose=verbose, shuffle=shuffle)
    end = timer()
    t = end - start
    print ("Training time: {}".format(t))
    ## evaluate the model
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    # print ('loss= ',_)
    print ("test accuracy {}".format(test_acc))
    t = 0
    # print ("reset time variable: t = {}".format(t))
    return model, test_acc

# RP with bagging, training data generation
# use training dataset only
## bootstrapping first then RP
def rp_data_gen (X_train, y_train, X_test, sample_portion=0.8, n_splits=15, thin_dim=1000, density='auto'):
    random_state = np.arange(1, n_splits+1)
    samples = round(sample_portion * len(X_train))
    # print ('samples number: {}'.format(samples))
    transformers = []
    X_train_thin_sets = []
    X_test_thin_sets = []
    X_valid_thin_sets = []
    y_train_thin_sets = []
    y_valid_thin_sets = []
    for n in range(n_splits):
        # select indexesm, ix means index
        ix = [i for i in range(len(X_train))]
        # resample returns new index for the new data
        train_ix = resample(ix, replace=True, n_samples=samples, random_state=random_state[n])
        test_ix = [x for x in ix if x not in train_ix]
        # select data
        trainX, trainy = X_train.iloc[train_ix], y_train.iloc[train_ix]
        # testing is not necessay here, can be used as validation set
        testX, testy = X_train.iloc[test_ix], y_train.iloc[test_ix]
        # projectoin matrix generation
        trans = random_projection.SparseRandomProjection(n_components=thin_dim, density= density, random_state= random_state[n])
        transformers.append(trans)
        a = trans.fit_transform(trainX)
        b = trans.fit_transform(testX) # validation for boostrapping
        c = trans.fit_transform(X_test) # X_test from the very beginning
        X_train_thin_sets.append(a)
        X_valid_thin_sets.append(b)
        X_test_thin_sets.append(c)
        y_train_thin_sets.append(trainy)
        y_valid_thin_sets.append(testy)
    return X_train_thin_sets, X_valid_thin_sets, X_test_thin_sets, y_train_thin_sets, y_valid_thin_sets

######
## first transform the same X_train N times with different RNG
## then bootstrapping
def rp_data_gen_reverse (X_train, y_train, X_test, sample_portion=0.8, n_splits=15, thin_dim=1000, density='auto'):
    random_state = np.arange(1, n_splits+1)
    samples = np.int(round(sample_portion * len(X_train)))
    
    # debugging print
    print (type(samples))
    print("samples={}".format(samples))
    # print ('samples number: {}'.format(samples))
    transformers = []
    X_train_thin_sets = []
    X_test_thin_sets = []
    X_valid_thin_sets = []
    y_train_thin_sets = []
    y_valid_thin_sets = []
    for n in range(n_splits):
        # RP matrix generation
        trans = random_projection.SparseRandomProjection(n_components=thin_dim, density= density, random_state= random_state[n])
        # transformers.append(trans)
        X_train_thin_temp = trans.fit_transform(X_train)
        X_test_thin_temp = trans.fit_transform(X_test)
    # # bootstrapping    
        # select indexesm, ix means index
        ix = [i for i in range(len(X_train_thin_temp))]
        # resample returns new index for the new data
        train_ix = resample(ix, replace=True, n_samples=samples, random_state=random_state[n])
        valid_ix = [x for x in ix if x not in train_ix]
        # select data
        X_train_thin, y_train_thin = X_train_thin_temp[train_ix], y_train.iloc[train_ix]
        # testing is not necessay here, can be used as validation set
        validX, validy = X_train_thin_temp[valid_ix], y_train.iloc[valid_ix]
        
        X_train_thin_sets.append(X_train_thin)
        y_train_thin_sets.append(y_train_thin)
        X_valid_thin_sets.append(validX)
        y_valid_thin_sets.append(validy)
        # only tranform, no bootstrapping for X_test
        X_test_thin_sets.append(X_test_thin_temp) 
    return X_train_thin_sets, X_valid_thin_sets, X_test_thin_sets, y_train_thin_sets, y_valid_thin_sets
######


# model fitting
# Use SGD by default
def model_fitting(X_train_thin_sets, y_train_thin_sets, X_valid_thin_sets, y_valid_thin_sets, n_splits = 15, avg_time=False, n_avg=50, verbose = False, classifier = 'sgd', input_dim = 1000, batch_size= 100, num_epochs= 1, num_iters = 1000, alpha=0.0001, early_stopping= True, eta=0, lr= 'optimal', penalty= 'l1', shuffle = True):
    scores, members = [],[]
    # split the training set to n_splits randomly get the model for each split
    for i in range(n_splits):
        trainX, trainy = X_train_thin_sets[i], y_train_thin_sets[i]
        testX, testy = X_valid_thin_sets[i], y_valid_thin_sets[i]
        # evaluate model
        # start = timer() ###
        if (classifier == 'mlp'):
            model, test_acc = evaluate_model_mlp(trainX, trainy, testX, testy, input_dim=input_dim, batch_size=batch_size, num_epochs=num_epochs, verbose=verbose, shuffle=shuffle)
        elif(classifier == 'sgd'):
            model, test_acc = evaluate_model_sgd_bs1(trainX, trainy, testX, testy, avg_time= avg_time, n_avg=n_avg, num_epochs= num_epochs, num_iters = num_iters, alpha=alpha, eta0=eta0, lr= lr)
        elif(classifier == 'gd'):
            model, test_acc = evaluate_model_sgd(trainX, trainy, testX, testy, avg_time= avg_time, n_avg=n_avg, early_stopping=early_stopping, eta0=eta, learning_rate=lr)
        else:
            print("WRONG CLASSIFIER")
            return -1
        # print (i)
        if (verbose):
            print('>%.3f' % test_acc)
        scores.append(test_acc)
        members.append(model)
    return scores, members

def float2fix2float(x, total, frac):
    assert frac < total, "frac should be smaller than total"
    x_fix = np.int32(x*(2** frac)) 
    x_fix = np.maximum(x_fix, -(2**(total-1))) 
    x_fix = np.minimum(x_fix, 2**(total-1)-1)
    x_fix = np.float32(x_fix/(2**frac)) 
    return x_fix