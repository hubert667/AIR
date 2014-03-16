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
#from src.python.comparison.ProbabilisticInterleave import ProbabilisticInterleave

#os.chdir("..")
#os.chdir("..")

def compare(vali_queries,clussifierPath,basic_ranker_path,click):
    
    
    clf = pickle.load( open( classifierPath ) )
    basic_ranker=pickle.load( open( basic_ranker_path ) )

    
    for i in range(len(weights)):
        weights[i]=weights[i].replace("[", "")
        weights[i]=weights[i].replace("]", "")
    
    
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

    first_win=0
    for i in range(iterations):
        q = training_queries[random.choice(training_queries.keys())]
        rankers[1]=classifier.getRanker(clf, basic_ranker,q)
        l, a = compar_interleave.interleave(rankers[x], rankers[y], q, 10)
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
    

    
    