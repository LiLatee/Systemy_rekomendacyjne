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

datasetFilesPath="hetrec2011-movielens-2k-v2/"
userRatedMoviesFileName="user_ratedmovies.dat"
DFUserRatedMovies=pd.read_csv(datasetFilesPath+userRatedMoviesFileName, header=0, delimiter="\t",usecols=['userID','movieID','rating'],nrows=10000)
#DFMovieActorsFileName="movie_actors.dat"
#DFMovieActors=pd.read_csv(datasetFilesPath+DFMovieActorsFileName, header=0, delimiter="\t",encoding='iso-8859-1',usecols=['movieID','actorID','ranking'],nrows=500)
#DFMovieActorsPivoted=DFMovieActors.pivot_table(index="movieID",columns="actorID",values="ranking")
#DFMovieActorsPivoted['movieID']=DFMovieActorsPivoted.index
#DFUserRatedMoviesWithMovieActors=pd.merge(DFUserRatedMovies,DFMovieActorsPivoted,on='movieID')
DFMovieGenresFileName="movie_genres.dat"
DFMovieGenres=pd.read_csv(datasetFilesPath+DFMovieGenresFileName, header=0, delimiter="\t")
DFMovieGenres['dummyColumn']=1
DFMovieGenresPivoted=DFMovieGenres.pivot_table(index="movieID",columns="genre",values="dummyColumn")
DFMovieGenresPivoted['movieID']=DFMovieGenresPivoted.index
DFUserRatedMoviesWithMovieGenres=pd.merge(DFUserRatedMovies,DFMovieGenresPivoted,on='movieID')
DFUserRatedMoviesWithMovieGenres=DFUserRatedMoviesWithMovieGenres.fillna(value=0)
#DFUserRatedMoviesWithMovieGenres=DFUserRatedMoviesWithMovieGenres.to_sparse()
DFUserRatedMoviesWithMovieGenres["rating"]=DFUserRatedMoviesWithMovieGenres["rating"]>DFUserRatedMoviesWithMovieGenres["rating"].mean()
DFUserRatedMoviesWithMovieGenres_y=DFUserRatedMoviesWithMovieGenres["rating"]
yTemp=DFUserRatedMoviesWithMovieGenres_y.values
y=np.where(yTemp,1,-1)
DFUserRatedMoviesWithMovieGenres_X=DFUserRatedMoviesWithMovieGenres.drop("rating",1)
#DFUserRatedMoviesWithMovieGenres_X=DFUserRatedMoviesWithMovieGenres_X.to_sparse()
#DFUserRatedMoviesWithMovieGenres_X2=pd.get_dummies(DFUserRatedMoviesWithMovieGenres_X["userID"],prefix=['userID'])
#DFUserRatedMoviesWithMovieGenres_X3=pd.get_dummies(DFUserRatedMoviesWithMovieGenres_X["movieID"],prefix=['movieID'])
#DFUserRatedMoviesWithMovieGenres_X.to_coo()
XUserID=DFUserRatedMoviesWithMovieGenres_X["userID"].values
XMovieID=DFUserRatedMoviesWithMovieGenres_X["movieID"].values
DFUserRatedMoviesWithMovieGenres_X=DFUserRatedMoviesWithMovieGenres_X.drop("userID",1)
DFUserRatedMoviesWithMovieGenres_X=DFUserRatedMoviesWithMovieGenres_X.drop("movieID",1)
XMovieGenres=csr_matrix(DFUserRatedMoviesWithMovieGenres_X.values)
OHE=OneHotEncoder()
#OHE.fit(XUserID)
XUserIDOHEncoded=OHE.fit_transform(XUserID.reshape(-1, 1))
XMovieIDOHEncoded=OHE.fit_transform(XMovieID.reshape(-1, 1))
X=hstack([XUserIDOHEncoded,XMovieIDOHEncoded,XMovieGenres])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
LRC=LogisticRegression()
LRC.fit(X_train,y_train)
y_probs=LRC.predict_proba(X_test)
y_pred=y_probs[:,1]
fpr, tpr, _ = roc_curve(y_test,y_pred)
roc_auc=auc(fpr, tpr)
print('LR AUC: {}'.format(roc_auc))