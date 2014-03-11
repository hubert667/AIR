import sys, random, ast
try:
    import include, copy, pickle
except:
    pass
import retrieval_system, environment, evaluation
import query as queryClass
from queryRankers import *

class QueryRanker():

    def __init__(self, path_train_dataset, path_test_dataset, feature_count_dataset, min_freq_count, iterations, rankers_per_query, click_model):
        ''' Constructor '''
        self.path_train = path_train_dataset
        self.path_test=path_test_dataset
        self.feature_count = feature_count_dataset    
        self.minFreqCount = min_freq_count
        self.iterationCount = iterations
        self.rankersPerQuery = rankers_per_query
        self.clickModel = click_model
    
    def queryRanker(self):
        #Extract the high frequency queries from the training_queries
        HighFreqQueries = []
        training_queries = queryClass.load_queries(self.path_train, self.feature_count)
        test_queries = queryClass.load_queries(self.path_test, self.feature_count)
        #loop through all queries in the training set
        for index in training_queries.get_qids():
            highQuery = training_queries.get_query(index)
            #only keep the frequent queries 
            if(len(highQuery.__labels__) > self.minFreqCount):
                HighFreqQueries.append(highQuery)    
        print "found "+ str(len(HighFreqQueries)) + " high frequency queries"

        #build the query-ranker dictionary
        BestRanker = queryRankers()

        user_model = environment.CascadeUserModel(self.clickModel)
        evaluation2 = evaluation.NdcgEval()
        #test_queries = query.load_queries(sys.argv[2], feature_count)
        print "Read in training and testing queries"
        #for every query learn the best ranker and save it to the dictionary
        iter=0
        for highQuery in HighFreqQueries:
            sys.stdout.write('\r'+str(iter*100/len(HighFreqQueries))+"%")
            sys.stdout.flush()
            iter=iter+1
            for i in xrange(self.rankersPerQuery):
                learner = retrieval_system.ListwiseLearningSystem(self.feature_count, '-w random -c comparison.ProbabilisticInterleave -r ranker.ProbabilisticRankingFunction -s 3 -d 0.1 -a 0.01')
                BestRanker.addInitRank(highQuery.get_qid(),learner.get_solution().w)
                q = highQuery
                for t in range(self.iterationCount):
                    l = learner.get_ranked_list(q)
                    c = user_model.get_clicks(l, q.get_labels())
                    s = learner.update_solution(c)
                    print evaluation2.evaluate_all(s, test_queries)
                BestRanker.add(highQuery.get_qid(),learner.get_solution().w)
                BestRanker.addList(highQuery.get_qid(),l)
                BestRanker.addEval(highQuery.get_qid(),e)
        #save the dictionary to a file ('bestRanker.p')
        paths=self.path_train.split('/')
        name=paths[1]
        pickle.dump(BestRanker, open( "QueryData/"+name+".data", "wb" ) )
        test = pickle.load( open( "QueryData/"+name+".data", "rb" ) )
        print test.query_ranker.keys()
        print test.query_ranker.values()


