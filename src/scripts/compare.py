from include import *
import yaml as yaml
import numpy as np
from comparison import *
import sys, random
try:
    import include
except:
    pass
import retrieval_system, environment, evaluation, query,comparison,ranker, classifier
from utils import get_class, split_arg_str, string_to_boolean
from comparison import *
import pickle
#from src.python.comparison.ProbabilisticInterleave import ProbabilisticInterleave

#os.chdir("..")
#os.chdir("..")

def compareSystems(vali_queries,classifierPath,basic_ranker_path,clust_data_path,click):
    
    print "-Loading Data-"
    clf = pickle.load( open( classifierPath ) )
    basic_ranker=pickle.load( open( basic_ranker_path ) )
    clusterData=pickle.load(open(clust_data_path))
    
    ranker_tie="random"
    feature_count=basic_ranker.feature_count
    ranker_args="3"
    arg_str=""
    sample_send="sample_unit_sphere"
    iterations=1000
    
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
        q = training_queries[random.choice(training_queries.keys())]
        rankers[1]=classifier.getRanker(clf, basic_ranker,q,clusterData)
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
    

    
    