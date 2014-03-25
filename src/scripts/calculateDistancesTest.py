import sys, random, ast, os
try:
    import include, copy, pickle
except:
    pass
import retrieval_system, environment, evaluation, query, ranker
from queryRankers import *
import numpy
import scipy.stats
import ranker as rankerClass
import query as queryClass

class calculateDistancesTest:

    def __init__(self, Path, feature_count, queriesPath):
        self.Path = Path
        self.queriesPath = queriesPath
        self.feature_count = feature_count
        self.listOrEucl = int(raw_input("Euclidean(0) or list(1) distance (answer with 0 or 1)?"))
        if(self.listOrEucl):
              print 'using list distance'
        else:
              print 'using euclidean distance'
        print "Reading in queries"
        self.training_queries = queryClass.load_queries(queriesPath, feature_count)
        print "Calculating distance"
        self.calculate()

    def euclidean_distance(self,a,b):
        aa=numpy.array(a)
        bb=numpy.array(b)
        dist = numpy.linalg.norm(aa-bb)
        return dist

    def list_distance(self,a,b,query):
        aa = str(a.tolist())
        aa = aa.replace('[', '').replace(']', '')
        bb = str(b.tolist())
        bb = bb.replace('[', '').replace(']', '')
        rankerA = ranker.ProbabilisticRankingFunction(['3'], "random", 64, aa, "sample_unit_sphere")
        rankerB = ranker.ProbabilisticRankingFunction(['3'], "random", 64, bb, "sample_unit_sphere")
        query = self.training_queries.get_query(query)
        rankerA.init_ranking(query)
        rankerB.init_ranking(query)
        docsA = rankerA.getDocs()
        docsB = rankerB.getDocs()
        docsA2 = [str(x.docid) for x in docsA]
        docsB2 = [str(x.docid) for x in docsB]
        tau, p_value = scipy.stats.kendalltau(docsA2,docsB2)
        #Values close to 1 indicate strong agreement, values close to -1 indicate strong disagreement
	#- range tau between 0 and 1
        tau = (tau + 1)/2
	#- invert values distance to get distance instead of agreement
	tau = 1 - tau       
        return tau

    def sumLists(self,dictOfList):
        res=0
        for localList in dictOfList:
            res=res+sum(localList)
        return res

    def lenLists(self,dictOfList):
        res=0
        for localList in dictOfList:
            res=res+len(localList)
        return res

    def calculate(self):        
        path = self.Path
        queryDistances={} # distances between pairs of rankers for the same query
        distancesForDifferentQ={} #distances between rankers for different query
        distancesFirst={} #distances between initial ranker and the ranker after training 
        queryRankers = pickle.load( open( path ) )
        prevRank=[]
        for query in queryRankers.query_ranker:
            rankers=queryRankers.query_ranker[query]  
            rankersBegin=queryRankers.query_init_ranker[query] 
            distances=[]
            distancesF=[]
            for i in range(len(rankers)):
                if(self.listOrEucl):
                      dF=self.list_distance(rankers[i],rankersBegin[i],query)
                else:
                      dF=self.euclidean_distance(rankers[i],rankersBegin[i])
                distancesF.append(dF)
                for j in range(i+1,len(rankers)):
                    if(self.listOrEucl):
                        dist=self.list_distance(rankers[i],rankers[j],query)
                    else:
                         dist=self.euclidean_distance(rankers[i],rankers[j])
                    distances.append(dist)
            if(len(prevRank)>0):
                if(self.listOrEucl):
                      dist=self.list_distance(prevRank,rankers[0],query)
                else:
                      dist=self.euclidean_distance(prevRank,rankers[0])
                distancesForDifferentQ[query]=dist
            queryDistances[query]=distances
            distancesFirst[query]=distancesF
            prevRank=rankers[0]        
        print "Average distance for the same query: "+str(self.sumLists(queryDistances.values())/float(self.lenLists(queryDistances.values())))
        print "Average distance for different queries: "+str(sum(distancesForDifferentQ.values())/float(len(distancesForDifferentQ)))
        print "Average distance ranker before and after learning: "+str(self.sumLists(distancesFirst.values())/float(self.lenLists(distancesFirst.values())))
