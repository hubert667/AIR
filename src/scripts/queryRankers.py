

class queryRankers:
    
    query_ranker={}
    
    def add(self,query,ranker):
        self.query_ranker[query]=ranker
        
    #def save(self,path):
        