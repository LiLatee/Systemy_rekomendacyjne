# -*- coding: utf-8 -*-
"""

@author: Szwabe
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


def getMovieAugmentationDictFromDF(movieAugmentationDict,DFMovieAugmentation,DFMovieAugmentationColumnLabel,DFMovieAugmentationPrefix,allFeaturesDict):
    DFMovieAugmentationDict=DFMovieAugmentation.to_dict("split")
    DFMovieAugmentationColumnsLabels=DFMovieAugmentationDict["columns"]
    DFMovieAugmentationList=DFMovieAugmentationDict["data"]
    movieIDIndex=DFMovieAugmentationColumnsLabels.index("movieID")
    augmentationColumnIndex=int(DFMovieAugmentationColumnsLabels.index(DFMovieAugmentationColumnLabel))
    for DFMovieAugmentationListItem in DFMovieAugmentationList:
        tempMovieID=int(DFMovieAugmentationListItem[movieIDIndex])
        tempFeatureValue=DFMovieAugmentationListItem[augmentationColumnIndex]
        if type(tempFeatureValue) is float:
            tempFeatureValue=int(tempFeatureValue)
#        aNewFeature=DFMovieAugmentationPrefix+"."+str(tempFeatureValue)
        aNewFeature=DFMovieAugmentationPrefix+"."+u"".join(tempFeatureValue).encode('utf-8')
        if not(aNewFeature in allFeaturesDict):
            allFeaturesDict[aNewFeature]=len(allFeaturesDict)
        if not (tempMovieID in list(movieAugmentationDict)):
            movieAugmentationDict[tempMovieID]=[]   
        movieAugmentationDict[tempMovieID].append(allFeaturesDict[aNewFeature])
    return movieAugmentationDict,allFeaturesDict

def batch_generator(X, y, batch_size):
    samples_per_epoch=X.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    X =  X[shuffle_index, :]
    y =  y[shuffle_index]
    while 1:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[index_batch,:].todense()
        y_batch = y[index_batch]
        counter += 1
        yield(np.array(X_batch),y_batch)
        if (counter < number_of_batches):
            np.random.shuffle(shuffle_index)
            counter=0

datasetFilesPath="hetrec2011-movielens-2k-v2/"
userRatedMoviesFileName="user_ratedmovies.dat"
DFUserRatedMovies=pd.read_csv(datasetFilesPath+userRatedMoviesFileName, header=0, delimiter="\t",usecols=['userID','movieID','rating'],nrows=500)
#DFUserRatedMovies=pd.read_csv(datasetFilesPath+userRatedMoviesFileName, header=0, delimiter="\t",usecols=['userID','movieID','rating'])
meanRating=DFUserRatedMovies["rating"].mean()
DFUserRatedMovies["rating"]=DFUserRatedMovies["rating"]>meanRating
DFMovieGenresFileName="movie_genres.dat"
DFMovieGenres=pd.read_csv(datasetFilesPath+DFMovieGenresFileName, header=0, delimiter="\t")
DFMovieActorsFileName="movie_actors.dat"
#DFMovieActors=pd.read_csv(datasetFilesPath+DFMovieActorsFileName, header=0, delimiter="\t",encoding='iso-8859-1',usecols=['movieID','actorID','ranking'],nrows=1000)
DFMovieActors=pd.read_csv(datasetFilesPath+DFMovieActorsFileName, header=0, delimiter="\t",encoding='iso-8859-1',usecols=['movieID','actorID','ranking'])
allFeaturesDict={}
movieAugmentationBySomethingDict={}
movieAugmentationBySomethingDict,allFeaturesDict=getMovieAugmentationDictFromDF(movieAugmentationBySomethingDict,DFMovieActors,"actorID","actor",allFeaturesDict)
movieAugmentationBySomethingDict,allFeaturesDict=getMovieAugmentationDictFromDF(movieAugmentationBySomethingDict,DFMovieGenres,"genre","genre",allFeaturesDict)
userRatedMovies=DFUserRatedMovies.to_dict("split")
userRatedMoviesData=userRatedMovies["data"]
userRatedMoviesColumns=userRatedMovies["columns"]
forCSRindptr=[0]
forCSRindices=[]
forCSRdata=[]
for userRatedMovieData in userRatedMoviesData:
    indicesOfFeaturesOfThisRow=[]
    tempUserID=userRatedMovieData[0]
    tempMovieID=userRatedMovieData[1]    
    for userRatedMoviesColumnIndex in range(2):
        DFMovieAugmentationPrefix=userRatedMoviesColumns[userRatedMoviesColumnIndex][:-2]
        tempFeatureValue=userRatedMovieData[userRatedMoviesColumnIndex]    
        aNewFeature=DFMovieAugmentationPrefix+"."+str(tempFeatureValue)
        if not(aNewFeature in allFeaturesDict):
            allFeaturesDict[aNewFeature]=len(allFeaturesDict)
        indicesOfFeaturesOfThisRow.append(allFeaturesDict[aNewFeature])
    if tempMovieID in movieAugmentationBySomethingDict:
        indicesOfFeaturesOfThisRow.extend(movieAugmentationBySomethingDict[tempMovieID])
    forCSRindices.extend(indicesOfFeaturesOfThisRow)
    for i in range(len(indicesOfFeaturesOfThisRow)):
        forCSRdata.append(1)
    forCSRindptr.append(len(forCSRindices))
X=csr_matrix((forCSRdata, forCSRindices, forCSRindptr))
y=DFUserRatedMovies["rating"].values
numberOfExperiments=1
resultsOfExperiments=[]
for currentExperimentNumber in range(numberOfExperiments):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print('Budowa modelu NN...')
    num_classes=2
    epochs = 3
    batch_size=1000
    model = Sequential()
#    model.add(Dense(512, input_shape=(len(allFeaturesDict),)))
    model.add(Dense(100, input_shape=(X_train.shape[1],)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
#    model.add(Dense(num_classes))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
#    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    #trainedModel=model.fit(X_train, y_train,epochs=epochs,verbose=1,validation_split=0.1)
    model.fit_generator(generator=batch_generator(X_train, y_train, batch_size),
                    nb_epoch=epochs, 
                    samples_per_epoch=X_train.shape[0])
    y_probs=model.predict(X_test.todense())
    fpr, tpr, _ = roc_curve(y_test,y_probs)
    roc_auc=auc(fpr, tpr)
    print('AUC: {}'.format(roc_auc))
    resultsOfExperiments.append(roc_auc)
    plt.plot(fpr,tpr)
    