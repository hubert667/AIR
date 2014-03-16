import sys, random, ast, os
try:
    import include, copy, pickle
except:
    pass
import retrieval_system, environment, evaluation, query
from queryRankers import *
import numpy
import scipy.stats

def euclidean_distance(a,b):
    aa=numpy.array(a)
    bb=numpy.array(b)
    dist = numpy.linalg.norm(aa-bb)
    return dist

def list_distance(a,b):
    aa = numpy.array(a)
    bb = numpy.array(b)
    tau, p_value = scipy.stats.kendalltau(aa,bb)
    dist = 1 - tau
    return dist

def sumLists(dictOfList):
    res=0
    for localList in dictOfList:
    res=res+sum(localList)
    return res

def lenLists(dictOfList):
    res=0
    for localList in dictOfList:
    res=res+len(localList)
    return res

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
    

print "Average distance for the same query: "+str(sumLists(queryDistances.values())/float(lenLists(queryDistances.values())))
print "Average distance for different queries: "+str(sum(distancesForDifferentQ.values())/float(len(distancesForDifferentQ)))
print "Average distance ranker before and after learning: "+str(sumLists(distancesFirst.values())/float(lenLists(distancesFirst.values())))



