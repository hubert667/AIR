from sklearn import svm
import sys, random, ast, os
try:
    import include, copy, pickle
except:
    pass
import retrieval_system, environment, evaluation,comparison
import query as queryClass
import ranker as rankerClass
from clusterData import *
import numpy as np


#path=sys.argv[1]

class Classifier:

    #rankerPath is never used in the code!!!
    def __init__(self, clusterPath, queriesPath, ranker):
        self.clusterDataPath = clusterPath
        self.testQueries = queriesPath
        self.rankerPath = None

    def Train(self):
        
        print "Loading Data"
        clusterData=pickle.load(open( self.clusterDataPath, "rb" ) )
        clustersData = pickle.load( open( self.clusterDataPath, "rb" ) )
        feature_count=len(clusterData.clusterToRanker[0][0])
        training_queries = queryClass.load_queries(self.testQueries, feature_count)
        #ranker = pickle.load( open( sys.argv[3], "rb" ) )
        
        testWeights=str(clusterData.clusterToRanker[0][0])
        testWeights=testWeights.replace("[", "")
        testWeights=testWeights.replace("]", "")
        weights = np.array([float(num) for num in testWeights.split(",")])
        print len(weights)
        ranker_tie="random"
        ranker_args="3"
        sample_send="sample_unit_sphere"

        ranker=rankerClass.ProbabilisticRankingFunction(ranker_args,
                                                ranker_tie,
                                                feature_count,
                                                sample=sample_send,
                                                init=testWeights)
        
        X=[]
        Y=[]
        print "Loading training objects"
        for qid in clusterData.queryToCluster:
            query = training_queries.get_query(qid)
            ranker.init_ranking(query)
            docIds=ranker.get_ranking()
            for docId in docIds:
                features=query.get_feature_vector(docId)
                X.append(features)
                Y.append(clusterData.queryToCluster[qid])
            
        #X = [[0, 0], [1, 1]]
        #y = [0, 1]
        print "Training"
        clf = svm.SVC()
        #clf.fit(X, y) 
       
        if not os.path.exists("Classifier"):
            os.makedirs("Classifier")

        paths=self.clusterDataPath.split('/')
        name=paths[len(paths)-1]
        parts=name.split('.')
        name=parts[0]
        pickle.dump(clf, open( "Classifier/"+name+".data", "wb" ) )
        
        
       
    #print test.predict([[2., 2.]])

    #test= pickle.load( open("../../../QueryData/1000iterations/NP2004.data", "rb" ) )
    #print len(test.query_ranker.values()[0][0])
