from sklearn import svm
import sys, random, ast, os
try:
    import include, copy, pickle
except:
    pass
import retrieval_system, environment, evaluation, query
from clusterData import *
import numpy


#path=sys.argv[1]


clustersData = pickle.load( open( sys.argv[2], "rb" ) )
training_queries = query.load_queries(sys.argv[1], feature_count)
ranker = pickle.load( open( sys.argv[3], "rb" ) )
X=[]
for qid in clusterData.queryToCluster:
    query = training_queries.get_query(qid)
    ranker.init_ranking(self, query)
    docIds=ranker.get_ranking()
    for docId in docIds:
        features=query.get_feature_vector(docId)
        X.append(features)
        Y.append(clusterData.queryToCluster[qid])
    
#X = [[0, 0], [1, 1]]
#y = [0, 1]
clf = svm.SVC()
clf.fit(X, y) 
pickle.dump(clf, open( "classifier.data", "wb" ) )
test = pickle.load( open( "classifier.data", "rb" ) )
#print test.predict([[2., 2.]])


