

class queryRankers:
    
    def __init__(self):
        self.query_ranker = {}  # an instance attribute
    
    def add(self,query,ranker):
        if query in self.query_ranker.keys():
                self.query_ranker[query].append(ranker)
        else:
            self.query_ranker[query]=[ranker]
        
  
        
        
  
        