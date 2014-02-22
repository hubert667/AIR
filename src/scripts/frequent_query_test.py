import sys, random
try:
    import include
except:
    pass
import retrieval_system, environment, evaluation, query
import os
from queryRankers import *
import pickle

#os.chdir("..")
#os.chdir("..")
#os.chdir("..")
feature_count=136
rankerDict=queryRankers()

#feature_count=64
learner = retrieval_system.ListwiseLearningSystem(feature_count, '-w random -c comparison.ProbabilisticInterleave -r ranker.ProbabilisticRankingFunction -s 3 -d 0.1 -a 0.01')
user_model = environment.CascadeUserModel('--p_click 0:0.0,1:1 --p_stop 0:0.0,1:0.0')
evaluation = evaluation.NdcgEval()
training_queries = query.load_queries(sys.argv[1], feature_count)
query_freq={}
for train in training_queries:
    
    if(len(train.__labels__) in query_freq):
        query_freq[len(train.__labels__)]=query_freq[len(train.__labels__)]+1
    else:
        query_freq[len(train.__labels__)]=1    
print query_freq                                 
test_queries = query.load_queries(sys.argv[2], feature_count)
#while True:
q = training_queries[random.choice(training_queries.keys())]
l = learner.get_ranked_list(q)
c = user_model.get_clicks(l, q.get_labels())
s = learner.update_solution(c)
print evaluation.evaluate_all(s, test_queries)
rankerDict.add(q,s.w)
pickle.dump( rankerDict, open( "save.p", "wb" ) )
rankerDict = pickle.load( open( "save.p", "rb" ) )
print rankerDict.query_ranker
