import sys, random, ast
try:
    import include, copy, pickle
except:
    pass
import retrieval_system, environment, evaluation, query
from queryRankers import *

#set parameters:
feature_count = 64 	#136 for MS-dataset
minFreqCount = 100 	#derived from histogram
iterationCount = 1000	#1000 to 10000 should be enough
rankersPerQuery = 20

#Extract the high frequency queries from the training_queries
HighFreqQueries = []
training_queries = query.load_queries(sys.argv[1], feature_count)
#loop through all queries in the training set
for index in training_queries.get_qids():
	query = training_queries.get_query(index)
	#only keep the frequent queries 
	if(len(query.__labels__) > minFreqCount):
		HighFreqQueries.append(query)	
print "found "+ str(len(HighFreqQueries)) + " high frequency queries"

#build the query-ranker dictionary
BestRanker = queryRankers()

user_model = environment.CascadeUserModel('--p_click 0:0.0,1:1 --p_stop 0:0.0,1:0.0')
#evaluation = evaluation.NdcgEval()
#test_queries = query.load_queries(sys.argv[2], feature_count)
print "Read in training and testing queries"

#for every query learn the best ranker and save it to the dictionary
for query in HighFreqQueries:
	for i in xrange(rankersPerQuery):
		learner = retrieval_system.ListwiseLearningSystem(feature_count, '-w random -c comparison.ProbabilisticInterleave -r ranker.ProbabilisticRankingFunction -s 3 -d 0.1 -a 0.01')
		q = query
		for t in range(iterationCount):
			l = learner.get_ranked_list(q)
			c = user_model.get_clicks(l, q.get_labels())
			s = learner.update_solution(c)
			#print evaluation.evaluate_all(s, test_queries)
		BestRanker.add(query.get_qid(),learner.get_solution().w)
#save the dictionary to a file ('bestRanker.p')
pickle.dump(BestRanker, open( "QueryData/NP2004QueryRankers.data", "wb" ) )
test = pickle.load( open( "QueryData/NP2004QueryRankers.data", "rb" ) )
print test.query_ranker.keys()
print test.query_ranker.values()
