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
        self.rankerPath = ranker

    def Train(self):
        
        print "Loading Data"
        clusterData=pickle.load(open( self.clusterDataPath, "rb" ) )
        feature_count=len(clusterData.clusterToRanker[0][0])
        training_queries = queryClass.load_queries(self.testQueries, feature_count)
        ranker=pickle.load( open( self.rankerPath ) )
        
        """
        testWeights=str(clusterData.clusterToRanker[0][0])
        testWeights=testWeights.replace("[", "")
        testWeights=testWeights.replace("]", "")
        weights = np.array([float(num) for num in testWeights.split(",")])
        ranker_tie="random"
        ranker_args="3"
        sample_send="sample_unit_sphere"

        ranker=rankerClass.ProbabilisticRankingFunction(ranker_args,
                                                ranker_tie,
                                                feature_count,
                                                sample=sample_send,
                                                init=testWeights)
        """
        X=[]
        Y=[]
        max=100 #max number of docs in the ranking 

    #print clusterData.queryToCluster.keys()
    #print training_queries.keys()
        print "Loading training objects"
        for qid in clusterData.queryToCluster:
            query = training_queries.get_query(qid)
            ranker.init_ranking(query)
            docIds=ranker.get_ranking()
            iter=0
            for docId in docIds:
                if iter>max:
                    break
                features=query.get_feature_vector(docId)
                X.append(features)
                Y.append(clusterData.queryToCluster[qid][iter])
                
                iter=iter+1
            
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
        
def getRanker(clf, basic_ranker,query,clusterData):
        
        max=10
        basic_ranker.init_ranking(query)
        docIds=basic_ranker.get_ranking()
        i=0
        results={}
        for docId in docIds:
            if i>max:
                break
            i=i+1
            features=query.get_feature_vector(docId)
            X=features
            y=clf.predict(features)
            y=y[0]
            if y in results:
                results[y]=results[y]+1
            else:
                results[y]=1
            
        found_max=0
        arg_max=0
        for k in results:
            if results[k]>found_max:
                found_max=results[k]
                arg_max=k
                
        rankerVec=clusterData.clusterToRanker[arg_max][0]
        
        ranker_tie="random"
        feature_count=len(rankerVec)
        ranker_args="3"
        arg_str=""
        sample_send="sample_unit_sphere"
        iterations=100
        
        testWeights=str(rankerVec)
        testWeights=testWeights.replace("[", "")
        testWeights=testWeights.replace("]", "")
        
        resultRanker=ranker.ProbabilisticRankingFunction(ranker_args,
                                                ranker_tie,
                                                feature_count,
                                                sample=sample_send,
                                                init=testWeights)
        
        return resultRanker
        
#clusterPath = "../../../ClusterData/"+"letor"+".data"
#path_train = '../../../Datasets/LETORConcat/2004Concat/Fold1/train.txt'
#ranker path is not used in the current classifier code...
#rankerPath = None
#C = Classifier(clusterPath, path_train, rankerPath)
#C.Train()