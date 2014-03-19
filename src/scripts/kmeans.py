import random, pickle, os, sys
import numpy as np
from clusterData import *


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

    '''Gap statistics implementation from: http://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/'''

    '''To run from console use three arguments "bestRankersPickleFile.p" fromRangeK toRangeK  '''
    #gap statistic
    def cluster_points(self, X, mu):
        clusters  = {}
        for x in X:
            bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                        for i in enumerate(mu)], key=lambda t:t[1])[0]
            try:
                clusters[bestmukey].append(x)
            except KeyError:
                clusters[bestmukey] = [x]
        return clusters
     
    #gap statistic
    def reevaluate_centers(self, mu, clusters):
        newmu = []
        keys = sorted(clusters.keys())
        for k in keys:
            newmu.append(np.mean(clusters[k], axis = 0))
        return newmu
     
    #gap statistic
    def has_converged(self, mu, oldmu):
        return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])
     
    #gap statistic
    def find_centers(self, X, K):
        # Initialize to K random centers
        oldmu = random.sample(X, K)
        mu = random.sample(X, K)
        clusters = None
        while not self.has_converged(mu, oldmu):
            oldmu = mu
            # Assign all points in X to clusters
            clusters = self.cluster_points(X, mu)
            # Reevaluate centers
            mu = self.reevaluate_centers(oldmu, clusters)
        return(mu, clusters)


    #gap statisitc
    def init_board_gauss(self, N, k):
        n = float(N)/k
        X = []
        for i in range(k):
            c = (random.uniform(-1, 1), random.uniform(-1, 1))
            s = random.uniform(0.05,0.5)
            x = []
            while len(x) < n:
                a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
                # Continue drawing points from the distribution in the range [-1,1]
                if abs(a) < 1 and abs(b) < 1:
                    x.append([a,b])
            X.extend(x)
        X = np.array(X)[:N]
        return X

    #gap statistic
    def Wk(self, mu, clusters):
        K = len(mu)
        return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
                   for i in range(K) for c in clusters[i]])

    #gap statistic
    def bounding_box(self, X):
        xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
        ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
        return (xmin,xmax), (ymin,ymax)
     
    #gap statistic
    def gap_statistic(self, X):
        allClusters = []
        (xmin,xmax), (ymin,ymax) = self.bounding_box(X)
        # Dispersion for real distribution
        ks = range(self.fromK, self.toK)
        Wks = np.zeros(len(ks))
        Wkbs = np.zeros(len(ks))
        sk = np.zeros(len(ks))
        for indk, k in enumerate(ks):
            mu, clusters = self.find_centers(X,k)
            Wks[indk] = np.log(self.Wk(mu, clusters))
            allClusters.append(clusters)
            # Create B reference datasets
            B = 10
            BWkbs = np.zeros(B)
            for i in range(B):
                Xb = []
                for n in range(len(X)):
                    Xb.append([random.uniform(xmin,xmax),
                              random.uniform(ymin,ymax)])
                Xb = np.array(Xb)
                mu, clusters = self.find_centers(Xb,k)
                BWkbs[i] = np.log(self.Wk(mu, clusters))
            Wkbs[indk] = sum(BWkbs)/B
            sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
            print str(k)+' Clusters done.'
        sk = sk*np.sqrt(1+1/B)
        return(ks, Wks, Wkbs, sk, allClusters)


    def get_best_clusters(self, data):
        kList = [i for i in range(self.fromK, self.toK)]
        ks, logWks, logWkbs, sk, clustersForK = self.gap_statistic(data)
        gap = []
        gap = logWkbs - logWks
        indexTuple = np.where(gap==max(gap))   #itemindex = numpy.where(array==item)
        bestClustersIndex = indexTuple[0][0]
        self.bestK = str(bestClustersIndex+self.fromK)
        
        print self.bestK + ' set as best K value.'
        '''plt.plot([i for i in range(self.fromK, self.toK)],logWks)
        plt.plot([i for i in range(self.fromK, self.toK)],logWkbs)
        plt.plot([i for i in range(self.fromK, self.toK)],gap)
        plt.show()'''
        clusters = []
        for i in clustersForK[bestClustersIndex]:
            clusters.append(clustersForK[bestClustersIndex][i])
        return clusters


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
        print 'data', data
        print 'data shape', data.shape
        return data


    def runScript(self):#"bestRanker.p"  sys.argv[1]
        #commented out part is for test purposes
        #data = np.vstack((random(150,2) + np.array([.5,.5]),random(150,2), random(150,2) + np.array([2.5,2.5]), rand(150,2) + np.array([10.5,10.5])))
        data = self.getData()
        self.bestKClusterGroup1 = self.get_best_clusters(data) #list > list(cluster#) > np.array,np.array etc...
        self.bestKClusterGroup2 = []

        #converting list > list(cluster#) > np.array (ranker),np.array etc... to list > list(cluster#-->index of list) > normal list(ranker),list etc...
        for i in range(len(self.bestKClusterGroup1)):
            self.bestKClusterGroup2.append([])
            for j in range(len(self.bestKClusterGroup1[i])):
                self.bestKClusterGroup2[i].append(self.bestKClusterGroup1[i][j].tolist())

        clusterDataObject = clusterData()

        for i in range(len(self.bestKClusterGroup2)):
            #make object ---> dict[clusterNumber:int] = list of all rankers (where rankers are also lists)
            clusterDataObject.clusterToRanker[i] = self.bestKClusterGroup2[i]
            print type(clusterDataObject.clusterToRanker[i]), len(clusterDataObject.clusterToRanker[i])


        #make object ---> dict[queryID:string] = list of cluster numbers as ints
        for i in clusterDataObject.clusterToRanker:
            for j in clusterDataObject.clusterToRanker[i]:
                for k in self.queryRankerDict:
                    if type(self.queryRankerDict[k]) == list:
                        for l in self.queryRankerDict[k]:
                            if l.tolist() == j:
                                if k in clusterDataObject.queryToCluster.keys():
                                    clusterDataObject.queryToCluster[k].append(i)
                                else:
                                    clusterDataObject.queryToCluster[k] = [i]
                    elif self.queryRankerDict[k].tolist() == j:
                        clusterDataObject.queryToCluster[k] = i  

        '''for i in clusterDataObject.queryToCluster:
            print i, clusterDataObject.queryToCluster[i]
            
        for i in clusterDataObject.clusterToRanker:
            print i, clusterDataObject.clusterToRanker[i]'''

        if not os.path.exists("ClusterData"):
            os.makedirs("ClusterData")

        pickle.dump(clusterDataObject, open("ClusterData/"+self.dataset+".data", "wb"))
        #pickle.dump(clusterDataObject, open("ClusterData/"+self.dataset+" k"+self.bestK+".data", "wb"))
        #pickle.dump(clusterDataObject.queryToCluster, open( "ClusterData/queryToClusterDict.data", "wb" ) )
        #pickle.dump(clusterDataObject.clusterToRanker, open( "ClusterData/clusterToRankerDict.data", "wb" ) )
        
        
        '''print '-----------------Print output of one of the object files-----------------------'
        loadedFile = pickle.load( open( "ClusterData/clusterToRankerDict.data", "rb" ) )
        for i in loadedFile:
            print i
            #sys.exit()
            for j in loadedFile[i]:
                print j'''
        return clusterDataObject.queryToCluster, clusterDataObject.clusterToRanker
