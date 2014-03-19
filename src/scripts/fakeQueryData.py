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
from queryRankers import *
from queryFeatures import *



#path=sys.argv[1]

class Fake:

    #rankerPath is never used in the code!!!
    def __init__(self, dataset, queriesPath, ranker,feature_count):
        self.dataset = dataset
        self.testQueries = queriesPath
        self.feature_count=feature_count
        self.rankerPath = ranker

    def Save(self):
        
        print "Loading Data"
        
        training_queries = queryClass.load_queries(self.testQueries, self.feature_count)
        ranker=pickle.load( open( self.rankerPath ) )
        
        max=10 #max number of docs in the ranking 
        
        #print clusterData.queryToCluster.keys()
        #print training_queries.keys()
        BestRanker = queryFeatures()
        print "Loading training objects"
        i=0
        for query in training_queries:
            #print str(i*100/len(training_queries))+"%"
            i=i+1
            #query = training_queries.get_query(qid)
            ranker.init_ranking(query)
            docIds=ranker.get_ranking()
            iter=0
            for docId in docIds:
                if iter>max:
                    break
                iter=iter+1
                features=query.get_feature_vector(docId)
                BestRanker.add(query.get_qid(),features)
                #print features
                #BestRanker.addFeaturesToQid([float(i) for i in features],query.get_qid())

        
        pickle.dump(BestRanker, open( "QueryData/"+self.dataset+".data", "wb" ) )
          

      
"""
dataset="letor"
path_train = '../../../Datasets/LETORConcat/2004Concat/Fold1/train.txt'
rankerPath = '../../../QueryData/generalRanker.data'
feature_count=64
C = Fake(dataset, path_train, rankerPath,feature_count)
C.Save()
"""
