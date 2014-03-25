from include import *
import yaml as yaml
import numpy as np
from comparison import *
import sys, random
try:
    import include
except:
    pass
import retrieval_system, environment, evaluation, query,comparison, classifier
from utils import get_class, split_arg_str, string_to_boolean
from comparison import *
import pickle
import ranker as rankerClass
import pylab as P
#from src.python.comparison.ProbabilisticInterleave import ProbabilisticInterleave

#os.chdir("..")
#os.chdir("..")

def compareSystems(vali_queries,classifierPath,basic_ranker_path,clust_data_path,data,click):
    
    print "-Loading Data-"
    clf = pickle.load( open( classifierPath ) )
    basic_ranker=pickle.load( open( basic_ranker_path ) )
    clusterData=pickle.load(open(clust_data_path))
    queryData=pickle.load(open(data))
    
    ranker_tie="random"
    feature_count=basic_ranker.feature_count
    ranker_args="3"
    arg_str=""
    sample_send="sample_unit_sphere"
    iterations=100
    
    rankers=[0]*2
    rankers[0]=basic_ranker
    
    
    user_model = environment.CascadeUserModel(click)
    training_queries = query.load_queries(vali_queries, feature_count)
    compar_interleave=ProbabilisticInterleave(None)

    first_win=0
    print "-Calculating-"
    
    for i in range(iterations):
        if i%(iterations/10)==0:
            print str(float(i)*100/float(iterations))+"%"
        q = training_queries.get_query(random.choice(queryData.query_ranker.keys()))
        
        test=queryData.query_ranker[q.get_qid()][0]
        testWeights=str(test)
        testWeights=testWeights.replace("[", "")
        testWeights=testWeights.replace("]", "")
        weights = np.array([float(num) for num in testWeights.split(",")])
        print len(weights)
        ranker_tie="random"
        ranker_args="3"
        sample_send="sample_unit_sphere"

        rankers[1]=rankerClass.ProbabilisticRankingFunction(ranker_args,
                                                ranker_tie,
                                                feature_count,
                                                sample=sample_send,
                                                init=testWeights)
        
        
        l, a = compar_interleave.interleave(rankers[0], rankers[1], q, 10)
        c = user_model.get_clicks(l, q.get_labels())
        o = compar_interleave.infer_outcome(l, a, c, q)
        if(o<0):
            first_win+=1
        elif(o==0):
            coin=random.random()
            if(coin>0.5):
                first_win+=1
    result_com=float(first_win)/float(iterations)
    print "Basic ranker win rate:"+ str(result_com)
    
def compareSystemsHist(vali_queries,classifierPath,basic_ranker_path,clust_data_path,data,click):
    
    print "-Loading Data-"
    clf = pickle.load( open( classifierPath ) )
    basic_ranker=pickle.load( open( basic_ranker_path ) )
    clusterData=pickle.load(open(clust_data_path))
    queryData=pickle.load(open(data))
    
    ranker_tie="random"
    feature_count=basic_ranker.feature_count
    ranker_args="3"
    arg_str=""
    sample_send="sample_unit_sphere"
    iterations=100
    
    rankers=[0]*2
    rankers[0]=basic_ranker
    
    
    user_model = environment.CascadeUserModel(click)
    training_queries = query.load_queries(vali_queries, feature_count)
    compar_interleave=ProbabilisticInterleave(None)

    print "-Calculating-"
    
    ii=0


    results=[]
    for qid in queryData.query_ranker.keys():
        print str(float(ii)*100/float(len(queryData.query_ranker.keys())))+"%"
        ii+=1
        q=training_queries.get_query(qid)
        for val in queryData.query_ranker[qid]:
            test=val
            #test=queryData.query_ranker[q][0]
            testWeights=str(test.tolist())
            testWeights=testWeights.replace("[", "")
            testWeights=testWeights.replace("]", "")
            #weights = np.array([float(num) for num in testWeights.split(",")])
            #print len(weights)
            ranker_tie="random"
            ranker_args="3"
            sample_send="sample_unit_sphere"
    
            rankers[1]=rankerClass.ProbabilisticRankingFunction(ranker_args,
                                                    ranker_tie,
                                                    feature_count,
                                                    sample=sample_send,
                                                    init=testWeights)
           
            second_win=0
            for i in range(iterations):
                #q = training_queries.get_query(random.choice(training_queries.keys()))          
                l, a = compar_interleave.interleave(rankers[0], rankers[1], q, 10)
                c = user_model.get_clicks(l, q.get_labels())
                o = compar_interleave.infer_outcome(l, a, c, q)
                if(o>0):
                    second_win+=1
                elif(o==0):
                    coin=random.random()
                    if(coin>0.5):
                        second_win+=1
            result_com=float(second_win)/float(iterations)
            results.append(result_com)

    g=P.hist(results, bins = 10)
    P.show(g)
    
def compareSystemsHistDifferentQ(vali_queries,classifierPath,basic_ranker_path,clust_data_path,data,click):
    
    print "-Loading Data-"
    clf = pickle.load( open( classifierPath ) )
    basic_ranker=pickle.load( open( basic_ranker_path ) )
    clusterData=pickle.load(open(clust_data_path))
    queryData=pickle.load(open(data))
    
    ranker_tie="random"
    feature_count=basic_ranker.feature_count
    ranker_args="3"
    arg_str=""
    sample_send="sample_unit_sphere"
    iterations=100
    
    rankers=[0]*2
    rankers[0]=basic_ranker
    
    
    user_model = environment.CascadeUserModel(click)
    training_queries = query.load_queries(vali_queries, feature_count)
    compar_interleave=ProbabilisticInterleave(None)

    print "-Calculating-"
    
    ii=0

    results=[]
    for qid in queryData.query_ranker.keys():
        print str(float(ii)*100/float(len(queryData.query_ranker.keys())))+"%"
        ii+=1
        #q=training_queries.get_query(qid)
        for val in queryData.query_ranker[qid]:
            test=val
            #test=queryData.query_ranker[q][0]
            testWeights=str(test.tolist())
            testWeights=testWeights.replace("[", "")
            testWeights=testWeights.replace("]", "")
            #weights = np.array([float(num) for num in testWeights.split(",")])
            #print len(weights)
            ranker_tie="random"
            ranker_args="3"
            sample_send="sample_unit_sphere"
    
            rankers[1]=rankerClass.ProbabilisticRankingFunction(ranker_args,
                                                    ranker_tie,
                                                    feature_count,
                                                    sample=sample_send,
                                                    init=testWeights)
           
            second_win=0
            for i in range(iterations):
                q = training_queries.get_query(random.choice(training_queries.keys()))          
                l, a = compar_interleave.interleave(rankers[0], rankers[1], q, 10)
                c = user_model.get_clicks(l, q.get_labels())
                o = compar_interleave.infer_outcome(l, a, c, q)
                if(o>0):
                    second_win+=1
                elif(o==0):
                    coin=random.random()
                    if(coin>0.5):
                        second_win+=1
            result_com=float(second_win)/float(iterations)
            results.append(result_com)
            
    g=P.hist(results, bins = 10)
    P.show(g)

    
    