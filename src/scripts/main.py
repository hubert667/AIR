import argparse, sys, random, ast
import retrieval_system, environment, evaluation, query
from queryRanker import QueryRanker
try:
    import include, copy, pickle
except:
    pass
from queryRankers import *


inputParser = argparse.ArgumentParser(description='Query-level personalisation')
info = {
'd' : 'Type of dataset, choice between "letor", "yandex" and "ms".',
'k' : 'Set the number of clusters manually (default = determined by gap statistic).'
}
        
# Required parameters
inputParser.add_argument('-d', '--dataset', type=str, help=info['d'], required=True, choices=['letor', 'yandex','ms'])
    
# Optional parameters
inputParser.add_argument('-k', '--k', type=int, help=info['k'], required=False) #No k = gap statistic
  
arguments = inputParser.parse_args()
     
print "-- Creating features --"
dataset = arguments.dataset

#Setting the variables for each dataset
if dataset == 'letor':
    feature_count = 64
    path_train = '../../../NP2004/Fold1/train.txt.gz'
    path_test = '../../../NP2004/Fold1/test.txt.gz'
    path_validate = '../../../NP2004/Fold1/vali.txt.gz'
if dataset == 'ms':
    feature_count = 136
    print '!!! only using Fold 1 data right now !!!'
    path_train = '../../../MS-datasets/Fold1/train.txt'
    path_test = '../../../NP2004/Fold1/train.txt'
    path_validate = '../../../NP2004/Fold1/train.txt'
if dataset == 'yandex':
    feature_count = 136
    path_train = '../../../imat2009-datasets/imat2009_learning3.txt'
    path_test = '../../../imat2009-datasets/imat2009_test3.txt'
    path_validate = None
    
print 'calculating rankers for the', dataset, 'dataset'
Q = QueryRanker(path_train, feature_count)
Q.queryRanker()
 
print "-- Creating clusters --"
print "TODO"
print "-- Classification --"
print "TODO"
k = arguments.k
if k == None:
    k = 'gap statistic'
print 'k =', k
print "-- Finished! --"
