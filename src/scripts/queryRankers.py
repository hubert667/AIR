

class queryRankers:
    
    #query_ranker={}
    
    def __init__(self):
        self.query_ranker = {}  # an instance attribute
    
    def add(self,query,ranker):
        self.query_ranker[query]=ranker
        
  
        