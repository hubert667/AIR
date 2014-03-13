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
import numpy


#path=sys.argv[1]


def Train(clusterDataPath,testQueries,rankerPath):
    
    print "Loading Data"
    clusterData=pickle.load(open( clusterDataPath, "rb" ) )
    clustersData = pickle.load( open( clusterDataPath, "rb" ) )
    feature_count=len(clusterData.clusterToRanker[0][0])
    training_queries = queryClass.load_queries(testQueries, feature_count)
    #ranker = pickle.load( open( sys.argv[3], "rb" ) )
    
    testWeights=str(clusterData.clusterToRanker[0][0])
    testWeights=testWeights.replace("[", "")
    testWeights=testWeights.replace("]", "")
    weights = numpy.array([float(num) for num in testWeights.split(",")])
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
        ranker.init_ranking( query)
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

    paths=clusterDataPath.split('/')
    name=paths[len(paths)-1]
    parts=name.split('.')
    name=parts[0]
    pickle.dump(clf, open( "Classifier/"+name+".data", "wb" ) )
    
    
    
#print test.predict([[2., 2.]])

clusterPath="../../../ClusterData/NP2004.data"
queriesPath='../../../Datasets/NP2004/Fold1/train.txt.gz'
ranker=None

#test= pickle.load( open("../../../QueryData/1000iterations/NP2004.data", "rb" ) )
#print len(test.query_ranker.values()[0][0])

Train(clusterPath,queriesPath,ranker)
