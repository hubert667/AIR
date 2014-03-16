import argparse, sys, random, ast, os, sys
import retrieval_system, environment, evaluation, comparison
import query as queryClass
import ranker as rankerClass
import numpy as np
from kmeans import KMeans
from queryRanker import QueryRanker
from classifier import Classifier
try:
    import include, copy, pickle
except:
    pass
from queryRankers import *
from clusterData import *
from sklearn import svm

inputParser = argparse.ArgumentParser(description='Query-level personalisation')
info = {
'd' : 'Type of dataset, choice between "letor", "yandex" and "ms"',
'r' : 'Run learning to rank, clustering, classification, clustering + classify, compare or all ',
'i' : 'Set the number of iterations (default = 1000)',
'm' : 'The mimimum frequency count for queries (default = 200)',
'rq' : 'The rankers per query (default = 5)',
'fk' : 'Minimal number of clusters',
'tk' : 'Max number of clusters'
}
        
### Required parameters
inputParser.add_argument('-d', '--dataset', type=str, help=info['d'], required=True, choices=['letor', 'yandex','ms'])
inputParser.add_argument('-r', '--run', type=str, help=info['r'], required=True, choices=['learn', 'cluster', 'classify', 'clusterclassify', 'compare', 'all'])

    
### Optional parameters
#min and max (default 2 & 5)
inputParser.add_argument('-fk', '--fromrangek', type=int, help=info['fk'], default=2, required=False)
inputParser.add_argument('-tk', '--torangek', type=int, help=info['tk'], default=5, required=False)

#1000 to 10000 should be enough
inputParser.add_argument('-i', '--iterations', type=int, help=info['i'], default=1000, required=False) 

#derived from histogram
inputParser.add_argument('-m', '--minfreqcount', type=int, help=info['m'], default=200, required=False)

inputParser.add_argument('-rq', '--rankersperquery', type=int, help=info['rq'], default=5, required=False) 
arguments = inputParser.parse_args()

        
dataset = arguments.dataset
#info
print 'Using the', arguments.dataset, 'dataset'
print 'Going to run', arguments.run
if arguments.run == 'learn' or arguments.run == 'all':
    print 'Minimal frequency count:', arguments.minfreqcount
    print 'Iterations:', arguments.iterations
    print 'Rankers per query:', arguments.rankersperquery
if arguments.run == 'cluster' or arguments.run == 'clusterclassify' or  arguments.run == 'all':
    print 'Min number of clusters:', arguments.fromrangek
    print 'Max number of clusters:', arguments.torangek

#Setting the variables for each dataset
if dataset == 'letor':
    feature_count = 64
    path_train = 'Datasets/LETORConcat/2004Concat/Fold1/train.txt'
    path_test = 'Datasets/LETORConcat/2004Concat/Fold1/test.txt'
    path_validate = 'Datasets/LETORConcat/2004_np_dataset/Fold1/vali.txt.gz'
    click = '--p_click 0:0.0,1:1 --p_stop 0:0.0,1:0.0'
if dataset == 'ms':
    feature_count = 136
    print '!!! only using Fold 1 data right now !!!'
    path_train = 'Datasets/MS-datasets/Fold1/train.txt'
    path_test = 'Datasets/MS-datasets/Fold1/test.txt'
    path_validate = 'Datasets/MS-datasets/Fold1/vali.txt'
    click = '--p_click 0:0.0,1:0.2,2:0.4,3:0.8,4:1.0 --p_stop 0:0.0,1:0.0,2:0.0,3:0.0,4:0.0'
if dataset == 'yandex':
    feature_count = 245
    path_train = 'Datasets/imat2009-datasets/imat2009_learning3.txt'
    path_test = 'Datasets/imat2009-datasets/imat2009_test3.txt'
    path_validate = None
    click = '--p_click 0:0.0,1:0.2,2:0.4,3:0.8,4:1.0 --p_stop 0:0.0,1:0.0,2:0.0,3:0.0,4:0.0'


if arguments.run == 'learn' or arguments.run == 'all':
    print "-- Learning to Rank --"
    Q = QueryRanker(path_train, path_test, feature_count, arguments.minfreqcount, arguments.iterations, arguments.rankersperquery, click, dataset)
    Q.queryRanker()

if arguments.run == 'cluster' or arguments.run == 'clusterclassify' or arguments.run == 'all': 
    print "-- Clustering --"
    bestRankersFile = 'QueryData/'+dataset+'.data'
    KM = KMeans(arguments.fromrangek, arguments.torangek, bestRankersFile, dataset)
    (queryToCluster, clusterToRanker) = KM.runScript()
    #print 'queryToCluster', queryToCluster

if arguments.run == 'classify' or arguments.run == 'clusterclassify' or arguments.run == 'all': 
    print "-- Classification --"
    clusterPath = "ClusterData/"+dataset+".data"
    #ranker path is not used in the current classifier code...
    rankerPath = "QueryData/generalRanker.data"
    C = Classifier(clusterPath, path_train, rankerPath)
    C.Train()
    
if arguments.run == 'compare' : 
    print "-- Comparison --"
    classifierPath = "Classifier/"+dataset+".data"
    basic_ranker_path="QueryData/generalRanker.data"
    compare(path_validate,classifierPath,basic_ranker_path,click)
    
print "-- Finished! --"
