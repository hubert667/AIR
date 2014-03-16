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
        X=np.array(X)
        Y=np.array(Y)
        print "Training"
        clf = svm.SVC()
        clf.fit(X, Y) 
       
        if not os.path.exists("Classifier"):
            os.makedirs("Classifier")

        paths=self.clusterDataPath.split('/')
        name=paths[len(paths)-1]
        parts=name.split('.')
        name=parts[0]
        pickle.dump(clf, open( "Classifier/"+name+".data", "wb" ) )
        
    def GetRanker(classifier, basic_ranker,query):
        
        max=10
        basic_ranker.init_ranking(query)
        docIds=basic_ranker.get_ranking()
        i=0
        results={}
        for docId in docIds:
            if i>max:
                break
            i=i+1
            features=query.clusterData.clusterToRankerget_feature_vector(docId)
            X=features
            y=clf.predict(features)
            if y in results:
                results[y]=resuts[y]+1
            else:
                results[y]=1
            
        found_max=0
        arg_max=0
        for k in results:
            if results[m]>found_max:
                found_max=result[k]
                arg_max=k
                
        rankerVec=clusterData.clusterToRanker[arg_max][0]
        
        ranker_tie="random"
        feature_count=len(rankerVec)
        ranker_args="3"
        arg_str=""
        sample_send="sample_unit_sphere"
        iterations=100
        
        resultRanker=ranker.ProbabilisticRankingFunction(ranker_args,
                                                ranker_tie,
                                                feature_count,
                                                sample=sample_send,
                                                init=rankerVec)
        
        return resultRanker
        
       
    #print test.predict([[2., 2.]])

    #test= pickle.load( open("../../../QueryData/1000iterations/NP2004.data", "rb" ) )
    #print len(test.query_ranker.values()[0][0])
