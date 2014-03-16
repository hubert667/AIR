import sys, random
try:
    import include, pickle
except:
    pass
import retrieval_system, environment, evaluation, query


learner = retrieval_system.ListwiseLearningSystem(64, '-w random -c comparison.ProbabilisticInterleave -r ranker.ProbabilisticRankingFunction -s 3 -d 0.1 -a 0.01')
user_model = environment.CascadeUserModel('--p_click 0:0.0,1:1 --p_stop 0:0.0,1:0.0')
evaluation = evaluation.NdcgEval()
training_queries = query.load_queries(sys.argv[1], 64)
test_queries = query.load_queries(sys.argv[2], 64)
i=0
for i in range(10):
    q = training_queries[random.choice(training_queries.keys())]
    l = learner.get_ranked_list(q)
    c = user_model.get_clicks(l, q.get_labels())
    s = learner.update_solution(c)
    print i
    i=i+1
    print evaluation.evaluate_all(s, test_queries)

pickle.dump(learner.ranker, open( "QueryData/"+"generalRanker"+".data", "wb" ) )

