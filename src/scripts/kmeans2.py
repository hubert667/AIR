import random, pickle, os, sys
import numpy as np
from clusterData import *



queryRankerList = []
bestKClusterGroup = []
queryRankerDict = {}

'''Gap statistics implementation from: http://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/'''

'''To run from console use three arguments "bestRankersPickleFile.p" fromRangeK toRangeK  '''

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
 
def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])
 
 
def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)



def init_board_gauss(N, k):
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


def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])


def bounding_box(X):
    xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
    ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    return (xmin,xmax), (ymin,ymax)
 

def gap_statistic(X, fromK, toK):
    allClusters = []
    (xmin,xmax), (ymin,ymax) = bounding_box(X)
    # Dispersion for real distribution
    ks = range(fromK, toK)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    for indk, k in enumerate(ks):
        mu, clusters = find_centers(X,k)
        Wks[indk] = np.log(Wk(mu, clusters))
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
            mu, clusters = find_centers(Xb,k)
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
        print str(k)+' Clusters done.'
    sk = sk*np.sqrt(1+1/B)
    return(ks, Wks, Wkbs, sk, allClusters)


def get_best_clusters(data, fromK, toK):
    kList = [i for i in range(fromK, toK)]
    ks, logWks, logWkbs, sk, clustersForK = gap_statistic(data,fromK,toK)
    gap = []
    gap = logWkbs - logWks
    indexTuple = np.where(gap==max(gap))   #itemindex = numpy.where(array==item)
    bestClustersIndex = indexTuple[0][0]
    print str(bestClustersIndex+fromK)+' set as best K value.'
    '''plt.plot([i for i in range(fromK, toK)],logWks)
    plt.plot([i for i in range(fromK, toK)],logWkbs)
    plt.plot([i for i in range(fromK, toK)],gap)
    plt.show()'''
    clusters = []
    for i in clustersForK[bestClustersIndex]:
        clusters.append(clustersForK[bestClustersIndex][i])
    return clusters


def getData(bestRankersFile):
    loadedFile = pickle.load( open( bestRankersFile, "rb" ) ) #dict-->print i, test.query_ranker[i]
    global queryRankerDict
    global queryRankerList

    for i in loadedFile.query_ranker.keys():
        queryRankerDict[i] = loadedFile.query_ranker[i]
    print len(queryRankerDict)
    for i in queryRankerDict.keys():
        if type(queryRankerDict[i]) == list:
            for j in queryRankerDict[i]:
                queryRankerList.append(j)
        else:
            queryRankerList.append(queryRankerDict[i])
    data = np.array(queryRankerList)
    print data
    print data.shape
    return data


def runScript(bestRankersFile, frK, tK):#"bestRanker.p"  sys.argv[1]
    fromK = int(frK)
    toK = int(tK)+1
    global bestKClusterGroup, queryRankerList, queryRankerDict
    #commented out part is for test purposes
    #data = np.vstack((rand(150,2) + np.array([.5,.5]),rand(150,2), rand(150,2) + np.array([2.5,2.5]), rand(150,2) + np.array([10.5,10.5])))
    bestKClusterGroup1 = get_best_clusters(getData(bestRankersFile),fromK,toK) #list > list(cluster#) > np.array,np.array etc...
    bestKClusterGroup2 = []

    #converting list > list(cluster#) > np.array (ranker),np.array etc... to list > list(cluster#-->index of list) > normal list(ranker),list etc...
    for i in range(len(bestKClusterGroup1)):
        bestKClusterGroup2.append([])
        for j in range(len(bestKClusterGroup1[i])):
            bestKClusterGroup2[i].append(bestKClusterGroup1[i][j].tolist())

    clusterDataObject = clusterData()

    for i in range(len(bestKClusterGroup2)):
        #make object ---> dict[clusterNumber:int] = list of all rankers (where rankers are also lists)
        clusterDataObject.clusterToRanker[i] = bestKClusterGroup2[i]
        print type(clusterDataObject.clusterToRanker[i]), len(clusterDataObject.clusterToRanker[i])


    #make object ---> dict[queryID:string] = list of cluster numbers as ints
    for i in clusterDataObject.clusterToRanker:
        for j in clusterDataObject.clusterToRanker[i]:
            for k in queryRankerDict:
                if type(queryRankerDict[k]) == list:
                    for l in queryRankerDict[k]:
                        if l.tolist() == j:
                            if k in clusterDataObject.queryToCluster.keys():
                                clusterDataObject.queryToCluster[k].append(i)
                            else:
                                clusterDataObject.queryToCluster[k] = [str(i)]
                elif queryRankerDict[k].tolist() == j:
                    clusterDataObject.queryToCluster[k] = i  

    '''for i in clusterDataObject.queryToCluster:
        print i, clusterDataObject.queryToCluster[i]
        
    for i in clusterDataObject.clusterToRanker:
        print i, clusterDataObject.clusterToRanker[i]'''

    if not os.path.exists("ClusterData"):
        os.makedirs("ClusterData")

    paths=bestRankersFile.split('/')
    name=paths[len(paths)-1]
    parts=name.split('.')
    name=parts[0]
    pickle.dump(clusterDataObject, open( "ClusterData/"+name+".data", "wb" ) )

    
    
    '''print '-----------------Print output of one of the object files-----------------------'
    loadedFile = pickle.load( open( "../../../ClusterData/clusterToRankerDict.data", "rb" ) )
    for i in loadedFile:
        print i
        #sys.exit()
        for j in loadedFile[i]:
            print j'''
    return clusterDataObject.queryToCluster, clusterDataObject.clusterToRanker




#runScript('C:\Users\Spyros\Documents\EclipseWorkspace\Project AIR\QueryData/1000iterations\NP2004.data', 2,3)
runScript(sys.argv[1], sys.argv[2], sys.argv[3])