from sklearn import svm
import sys, random, ast, os
try:
    import include, copy, pickle
except:
    pass
import retrieval_system, environment, evaluation,comparison,ranker
import query as queryClass
import ranker as rankerClass
from clusterData import *
import numpy as np
from queryFeatures import *
from fakeQueryData import *
from fakeRankers import *
from kmeans import KMeans


def ClusterQueryDoc(dataset,rankerPath,feature_count, path_train_dataset, path_test_dataset, iterations, click_model, clusterData,queryDataPath,from_var,to_var):
    
   C = Fake(dataset, path_train_dataset, rankerPath,feature_count)
   C.Save()
   
   bestRankersFile = 'QueryData/'+dataset+'.data'
   KM = KMeans(from_var, to_var, bestRankersFile, dataset)
   (queryToCluster, clusterToRanker) = KM.runScript()
   
   g=GroupRanker(path_train_dataset,path_test_dataset,feature_count,iterations,click_model,dataset,clusterData,queryDataPath)
   g.groupRanker()
