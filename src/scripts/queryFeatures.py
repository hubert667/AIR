class queryFeatures:
    
    def __init__(self):
        self.query_ranker = {}  # key: query, value: list of q-d features
        self.ranker_query = {}  # key: features, value: query

    
    def add(self,query,ranker):

        if query in self.query_ranker.keys():
                self.query_ranker[query].append(ranker)
        else:
            self.query_ranker[query]=[ranker]
            
    def addFeaturesToQid(self,features,query):

        if query in self.ranker_query.keys():
                self.ranker_query[str(features)]=query
        else:
            self.ranker_query[str(features)]=query



        
  
        



