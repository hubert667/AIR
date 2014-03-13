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
queryDistances={} # distances between pairs of rankers for the same query
distancesForDifferentQ={} #distances between rankers for different query
distancesFirst={} #distances between initial ranker and the ranker after training 
queryRankers = pickle.load( open( path, "rbdist = numpy.linalg.norm(aa-bb)" ) )
prevRank=[]
for query in queryRankers.query_ranker:
    rankers=queryRankers.query_ranker[query]  
    rankersBegin=queryRankers.query_init_ranker[query] 
    distances=[]
    distancesF=[]
    for i in range(len(rankers)):
        dF=euclidean_distance(rankers[i],rankersBegin[i])
        distancesF.append(dF)
        for j in range(i+1,len(rankers)):
            dist=euclidean_distance(rankers[i], rankers[j])
            distances.append(dist)
    if(len(prevRank)>0):
        dist=euclidean_distance(prevRank,rankers[0])
        distancesForDifferentQ[query]=dist
    queryDistances[query]=distances
    distancesFirst[query]=distancesF
    prevRank=rankers[0]
    

print queryDistances.values()

