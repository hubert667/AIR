import sys, random, ast
try:
    import include, copy, pickle
except:
    pass
import retrieval_system, environment, evaluation
import query as queryClass
from queryRankers import *
import numpy as np

class GroupRanker():


    def __init__(self, path_train_dataset, path_test_dataset, feature_count_dataset, iterations, click_model, datasetType,clusterData,queryDataPath):

        ''' Constructor '''
        self.path_train = path_train_dataset
        self.path_test=path_test_dataset
        self.feature_count = feature_count_dataset    
        self.iterationCount = iterations
        self.clickModel = click_model
        self.dataset = datasetType
        self.clusterDataPath=clusterData
        self.queryData=queryDataPath

    
    def groupRanker(self):
        #Extract the high frequency queries from the training_queries
        clusterData=pickle.load(open( self.clusterDataPath, "rb" ) )
        queryData= self.queryData

        
        HighFreqQueries = []
        training_queries = queryClass.load_queries(self.path_train, self.feature_count)
        test_queries = queryClass.load_queries(self.path_test, self.feature_count)
        #loop through all queries in the training set
        

        #build the query-ranker dictionary
        BestRanker = queryRankers()

        user_model = environment.CascadeUserModel(self.clickModel)
        evaluation2 = evaluation.NdcgEval()
        #test_queries = query.load_queries(sys.argv[2], feature_count)
        print "Read in training and testing queries"
        #for every query learn the best ranker and save it to the dictionary
        iter=0
        learner=[0]*len(clusterData.clusterToRanker.keys())
        for cluster in clusterData.clusterToRanker:
            learner[cluster] = retrieval_system.ListwiseLearningSystem(self.feature_count, '-w random -c comparison.ProbabilisticInterleave -r ranker.ProbabilisticRankingFunction -s 3 -d 0.1 -a 0.01')  
        for t in range(self.iterationCount):
            q = training_queries[random.choice(training_queries.keys())]
            temp=(float(np.sum(clusterData.queryToCluster[q.get_qid()])))/(float(len(clusterData.queryToCluster[q.get_qid()])))
            temp=int(temp+0.5)
            cluster=temp
            #cluster=clusterData.queryToCluster[q.get_qid()][0]
            
            iter=iter+1
            if iter%200==0:
                print str(iter*100/self.iterationCount)+"%"
            l = learner[cluster].get_ranked_list(q)
            c = user_model.get_clicks(l, q.get_labels())
            s = learner[cluster].update_solution(c)
            #e = evaluation2.evaluate_all(s, test_queries)
        for cluster in clusterData.clusterToRanker:
             clusterData.clusterToRanker[cluster]=[learner[cluster].get_solution().w.tolist()]
      
            
        #save the dictionary to a file ('bestRanker.p')
        paths=self.path_train.split('/')
        name=paths[1]
        #pickle.dump(BestRanker, open( "QueryData/"+name+".data", "wb" ) )
        pickle.dump(clusterData, open( "ClusterData/"+self.dataset+".data", "wb" ) )

"""
dataset="letor"
path_train = 'Datasets/LETORConcat/2004Concat/Fold1/train.txt'
path_test = 'Datasets/LETORConcat/2004Concat/Fold1/test.txt'
queryDataPath='QueryData/'+dataset+'.data'
queryData=ranker=pickle.load( open( queryDataPath ) )
clusterPath='ClusterData/'+dataset+'.data'
click = '--p_click 0:0.0,1:1 --p_stop 0:0.0,1:0.0'
iterations=10
g=GroupRanker(path_train,path_test,64,iterations,click,dataset,clusterPath,queryData)
g.groupRanker()
"""
