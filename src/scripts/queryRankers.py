class queryRankers:
    
    def __init__(self):
        self.query_ranker = {}  # key: query, value: list of rankers after learning to rank
        self.query_init_ranker = {}  # key: query, value: list of rankers BEFORE learning to rank
        self.query_list={} # key: query, value: list of documents retrieved
        self.query_eval={} # key: query, value: list of evaluations after learning for every ranker
    
    def add(self,query,ranker):

        if query in self.query_ranker.keys():
                self.query_ranker[query].append(ranker)
        else:
            self.query_ranker[query]=[ranker]
            
    def addInitRank(self,query,ranker):

        if query in self.query_ranker.keys():
                self.query_ranker[query].append(ranker)
        else:
            self.query_ranker[query]=[ranker]
            
    def addList(self,query,list):

        if query in self.query_list.keys():
                self.query_list[query].append(list)
        else:
            self.query_list[query]=[list]
            
    def addEval(self,query,eval):

        if query in self.query_eval.keys():
                self.query_eval[query].append(eval)
        else:
            self.query_eval[query]=[eval]


        
  
        
