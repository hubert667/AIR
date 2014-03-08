import sys, random, ast, os
try:
    import include, copy, pickle
except:
    pass
import retrieval_system, environment, evaluation, query
from queryRankers import *
import numpy

def euclidean_distance(a,b):
    aa=numpy.array(a)
    bb=numpy.array(b)
    dist = numpy.linalg.norm(aa-bb)
    return dist

#os.chdir("..")
#os.chdir("..")
#os.chdir("..")

#path="QueryData/NP2004.data"
path=sys.argv[1]
queryDistances={}
queryRankers = pickle.load( open( path, "rb" ) )
for query in queryRankers.query_ranker:
    rankers=queryRankers.query_ranker[query]
    distances=[]
    for i in range(len(rankers)):
        for j in range(i+1,len(rankers)):
            dist=euclidean_distance(rankers[i], rankers[j])
            distances.append(dist)
    queryDistances[query]=distances
    
print queryDistances.keys()
print queryDistances.values()

