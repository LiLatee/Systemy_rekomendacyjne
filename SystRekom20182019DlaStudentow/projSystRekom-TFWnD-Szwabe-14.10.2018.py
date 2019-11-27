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

import tensorflow as tf
from absl import flags

import os
import itertools
import shutil
import copy

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
        aNewFeature=DFMovieAugmentationPrefix+"."+str(tempFeatureValue)
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

def train_input_fn(df):
    print("???",pd.Series(df["rating"].values[-10:]))
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: df[k].values for k in ["userID","movieID"]}),
        y=pd.Series(df["rating"].values),
    )

def tf_input_fn(X_as_df,y_as_df):
    return tf.estimator.inputs.pandas_input_fn(
        x=X_as_df,
        y=y_as_df,
        shuffle=True,
    )

def tf_eval_input_fn(X_as_df,y_as_df):
    return tf.estimator.inputs.pandas_input_fn(
        x=X_as_df,
        y=y_as_df,
        shuffle=False,
    )

def build_model_columns():
  movieID=tf.feature_column.categorical_column_with_hash_bucket('movieID', hash_bucket_size=10000)
  userID=tf.feature_column.categorical_column_with_hash_bucket('userID', hash_bucket_size=10000)
  base_columns = [movieID,userID]
  wide_columns = base_columns
  deep_columns = [
      tf.feature_column.embedding_column(movieID, dimension=100),
      tf.feature_column.embedding_column(userID, dimension=100),
      #tf.feature_column.categorical_column_with_identity
  ]
  return wide_columns, deep_columns


def build_estimator(model_dir):
    wide_columns, deep_columns = build_model_columns()
    #hidden_units = [100, 75, 50, 25]
    hidden_units = [2]
    # CPU is faster than GPU for a model including a 'wide' part.
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))
    return tf.estimator.DNNLinearCombinedRegressor(
    model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        #n_classes=2,
        #train_epochs=20,
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1),
        linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.1),
        #activation_fn=tf.nn.sigmoid,
        config=run_config)
#     return tf.estimator.LinearRegressor(
#         model_dir=model_dir,
#         feature_columns=wide_columns,
#         optimizer=tf.train.FtrlOptimizer(learning_rate=0.05),
#         config=run_config)


def build_estimator_of_given_type(model_dir, model_type):
  wide_columns, deep_columns = build_model_columns()
#  hidden_units = [100, 75, 50, 25]
  hidden_units = [2]
  # CPU is faster than GPU for a model including a 'wide' part.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))
  if model_type == 'wide':
    return tf.estimator.LinearRegressor(
        model_dir=model_dir,
        feature_columns=wide_columns,
        optimizer=tf.train.FtrlOptimizer(learning_rate=0.1),
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNRegressor(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1),
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedRegressor(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.1),
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1),
        config=run_config)



# def build_estimator(model_dir, model_type):
#   wide_columns, deep_columns = build_model_columns()
#   hidden_units = [100, 75, 50, 25]
#   # CPU is faster than GPU for a model including a 'wide' part.
#   run_config = tf.estimator.RunConfig().replace(
#       session_config=tf.ConfigProto(device_count={'GPU': 0}))
#   if model_type == 'wide':
#     return tf.estimator.LinearRegressor(
#         model_dir=model_dir,
#         feature_columns=wide_columns,
#         config=run_config)
#   elif model_type == 'deep':
#     return tf.estimator.DNNRegressor(
#         model_dir=model_dir,
#         feature_columns=deep_columns,
#         hidden_units=hidden_units,
#         config=run_config)
#   else:
#     return tf.estimator.DNNLinearCombinedRegressor(
#         model_dir=model_dir,
#         linear_feature_columns=wide_columns,
#         dnn_feature_columns=deep_columns,
#         dnn_hidden_units=hidden_units,
#         config=run_config)


def getLongTailOfDF(inputDF):
    outputDF=copy.deepcopy(inputDF)
    outputDF['countForMovies']=outputDF['rating'].groupby(outputDF['movieID']).transform('count')
    outputDF.sort_values(by=['countForMovies'])
    outputDF=outputDF.tail(int(outputDF.shape[0]/20))
    return outputDF


datasetFilesPath="hetrec2011-movielens-2k-v2/"
userRatedMoviesFileName="user_ratedmovies.dat"
#DFUserRatedMovies=pd.read_csv(datasetFilesPath+userRatedMoviesFileName, header=0, delimiter="\t",usecols=['userID','movieID','rating'],nrows=500)
DFUserRatedMovies=pd.read_csv(datasetFilesPath+userRatedMoviesFileName, header=0, delimiter="\t",usecols=['userID','movieID','rating'])
meanRating=DFUserRatedMovies["rating"].mean()
DFUserRatedMovies["rating"]=DFUserRatedMovies["rating"]>meanRating
DFUserRatedMovies["rating"]=DFUserRatedMovies["rating"].astype(int)
DFUserRatedMovies["movieID"]=DFUserRatedMovies["movieID"].astype(str)
DFUserRatedMovies["userID"]=DFUserRatedMovies["userID"].astype(str)
DFUserRatedMovies=getLongTailOfDF(DFUserRatedMovies)
y=DFUserRatedMovies["rating"]
X=DFUserRatedMovies.drop("rating",axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model_dir=os.getcwd()+"/model"
#flags.FLAGS.mark_as_parsed()
#model=build_estimator(flags_obj.model_dir,flags_obj.model_type)
shutil.rmtree(model_dir, ignore_errors=True)
#model=build_estimator(model_dir)
model_type="_"
#model_type="wide"
#model_type="deep"
model=build_estimator_of_given_type(model_dir,model_type)
model.train(input_fn=tf_input_fn(X_train,y_train),steps=100)
#evaluations=model.evaluate(input_fn=tf_eval_input_fn(X_train,y_train))
predictions=model.predict(input_fn=tf_eval_input_fn(X_test,y_test))
predictions_as_list_of_arrays=list(p["predictions"] for p in itertools.islice(predictions,None,None))
predictions_as_list=[]
for predictions_as_list_of_arrays_item in predictions_as_list_of_arrays:
    predictions_as_list.append(predictions_as_list_of_arrays_item[0])
fpr, tpr, _ = roc_curve(y_test.values,predictions_as_list)
roc_auc=auc(fpr, tpr)
print('TF AUC: {}'.format(roc_auc))
LRC=LogisticRegression(solver="sag")
OHEUsers=OneHotEncoder(handle_unknown="ignore")
OHEMovies=OneHotEncoder(handle_unknown="ignore")
#OHE.fit(XUserID)
XUserID=X_train["userID"].values
XMovieID=X_train["movieID"].values
XUserIDOHEncoded=OHEUsers.fit_transform(XUserID.reshape(-1, 1))
XMovieIDOHEncoded=OHEMovies.fit_transform(XMovieID.reshape(-1, 1))
X_train=hstack([XUserIDOHEncoded,XMovieIDOHEncoded])
LRC.fit(X_train,y_train)
XUserID=X_test["userID"].values
XMovieID=X_test["movieID"].values
XUserIDOHEncoded=OHEUsers.transform(XUserID.reshape(-1, 1))
XMovieIDOHEncoded=OHEMovies.transform(XMovieID.reshape(-1, 1))
X_test=hstack([XUserIDOHEncoded,XMovieIDOHEncoded])
y_probs=LRC.predict_proba(X_test)
y_pred=y_probs[:,1]
fpr, tpr, _ = roc_curve(y_test,y_pred)
roc_auc=auc(fpr, tpr)
print('LR AUC: {}'.format(roc_auc))
