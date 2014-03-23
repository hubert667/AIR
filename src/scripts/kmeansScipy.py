import random, pickle, os, sys
import numpy as np
from clusterData import *
from scipy.cluster.vq import kmeans,vq,whiten


class KMeans:

    def __init__(self, fK, tK, filename, typeDataset):
        self.queryRankerList = []
        self.bestKClusterGroup = []
        self.queryRankerDict = {}
        self.fromK = fK
        self.toK = tK + 1
        self.bestRankersFile = filename
        self.bestK = 0
        self.dataset = typeDataset


    def getClusters(self, thedata):
        # data generation
        data = whiten(thedata)
        # computing K-Means with K = 2 (2 clusters)
        centroids,_ = kmeans(data,self.fromK)

        # assign each sample to a cluster
        idx,_ = vq(data,centroids)
        return idx

    def getData(self):
        loadedFile = pickle.load( open( self.bestRankersFile, "rb" ) ) #dict-->print i, test.query_ranker[i]

        for i in loadedFile.query_ranker.keys():
            self.queryRankerDict[i] = loadedFile.query_ranker[i]
        print len(self.queryRankerDict)
        for i in self.queryRankerDict.keys():
            if type(self.queryRankerDict[i]) == list:
                for j in self.queryRankerDict[i]:
                    self.queryRankerList.append(j)
            else:
                self.queryRankerList.append(self.queryRankerDict[i])
        data = np.array(self.queryRankerList)

        return data


    def runScript(self):#"bestRanker.p"  sys.argv[1]
        #commented out part is for test purposes
        #data = np.vstack((random(150,2) + np.array([.5,.5]),random(150,2), random(150,2) + np.array([2.5,2.5]), rand(150,2) + np.array([10.5,10.5])))
        data = self.getData()
        dataToClusters = self.getClusters(data) #list > list(cluster#) > np.array,np.array etc...
        dataToClusters = list(dataToClusters)
        
        clusterDataObject = clusterData()
        data = list(data)
        #make object ---> dict[clusterNumber:int] = list of all rankers (where rankers are also lists)
        for i in range(len(dataToClusters)):
            if not dataToClusters[i] in clusterDataObject.clusterToRanker.keys():
                clusterDataObject.clusterToRanker[dataToClusters[i]] = [list(data[i])]
            else:
                clusterDataObject.clusterToRanker[dataToClusters[i]].append(list(data[i]))
                
        #make object ---> dict[queryID:string] = list of cluster numbers as ints
        for i in clusterDataObject.clusterToRanker:#for each cluster
            for j in clusterDataObject.clusterToRanker[i]:#for each ranker in cluster
                for q in self.queryRankerDict:#for each query
                    for r in self.queryRankerDict[q]:#for each ranker in query
                        if list(r) == j:#if ranker in query is equal to j (current ranker in cluster)
                            if q in clusterDataObject.queryToCluster:#if query key exists in dictionary
                                clusterDataObject.queryToCluster[q].append(i)
                            else:
                                clusterDataObject.queryToCluster[q] = [i]
                        
        
        for i in clusterDataObject.queryToCluster:
            print i, len(clusterDataObject.queryToCluster[i]), clusterDataObject.queryToCluster[i]
            
        for i in clusterDataObject.clusterToRanker:
            print i, len(clusterDataObject.clusterToRanker[i])#, clusterDataObject.clusterToRanker[i]  
     
        if not os.path.exists("ClusterData"):
            os.makedirs("ClusterData")

        pickle.dump(clusterDataObject, open("ClusterData/"+self.dataset+".data", "wb"))

        return clusterDataObject.queryToCluster, clusterDataObject.clusterToRanker

