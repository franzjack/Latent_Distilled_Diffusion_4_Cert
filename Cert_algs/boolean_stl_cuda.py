import torch
cuda = True if torch.cuda.is_available() else False

device = 'cuda' if cuda else 'cpu'


class STLFormula:
    def __init__(self):
        pass

    def evaluate(self, signal):
        pass


class bAtomicPredicate(STLFormula):
    def __init__(self, predicate, ind, thresh=0, lte=True):
        self.predicate = predicate
        self.ind = ind
        self.thresh = thresh
        self.lte = lte

    def evaluate(self, signal):
        #return torch.tensor([self.predicate(x) for x in signal], dtype=torch.float32)
        return self.predicate(signal,self.ind,self.thresh, self.lte)

class bAtomicPredicateNorm(STLFormula):
    def __init__(self, predicate, center, radius, lte=True):
        self.predicate = predicate
        self.center = center
        self.radius = radius
        self.lte = lte

    def evaluate(self, signal):
        #return torch.tensor([self.predicate(x) for x in signal], dtype=torch.float32)
        return self.predicate(signal,self.center,self.radius, self.lte)


class bAnd(STLFormula):
    def __init__(self, predicate1, predicate2):
        super().__init__()
        self.predicate1 = predicate1
        self.predicate2 = predicate2

    def evaluate(self, signal):
        # Evaluate both predicates
        n = signal.shape[0]
        result1 = self.predicate1(signal)
        result2 = self.predicate2(signal)


        return result1*result2

class bOr(STLFormula):
    def __init__(self, predicate1, predicate2):
        super().__init__()
        self.predicate1 = predicate1
        self.predicate2 = predicate2

    def evaluate(self, signal):
        n = signal.size(0)
        # Evaluate both predicates
        result1 = self.predicate1(signal)
        result2 = self.predicate2(signal)

        return torch.sign(result1+result2).to(device)

class bAlways(STLFormula):
    def __init__(self, predicate):
        self.predicate = predicate


    def evaluate(self, signal):
        n = signal.shape[0]
        bool_sign = self.predicate(signal)
        m = bool_sign.shape[1]

        res = torch.ones((n,1)).to(device)
        for i in range(m):
            res *= bool_sign[:,i:(i+1)].to(device)


        return res

    def evaluate_time(self, signal):
        n = signal.shape[0]
        bool_sign = self.predicate(signal)
        m = bool_sign.shape[1]


        results = bool_sign[:,-1:].to(device)
        res = bool_sign[:,-1:].clone().to(device)
        for i in range(m-2,-1,-1):

            res *= bool_sign[:,i:(i+1)].to(device)

            results = torch.cat((res.clone(),results),1).to(device)


        return results



class bEventually(STLFormula):
    def __init__(self, predicate):
        self.predicate = predicate

    def evaluate(self, signal):
        n = signal.shape[0]
        bool_sign = self.predicate(signal).to(device)

        m = bool_sign.shape[1]

        res = torch.zeros((n,1)).to(device)
        for i in range(m):
            res += bool_sign[:,i:(i+1)].to(device)


        return torch.sign(res).to(device)#.unsqueeze(1)

    def evaluate_time(self, signal):
        n = signal.shape[0]
        bool_sign = self.predicate(signal).to(device)
        m = bool_sign.shape[1]


        results = bool_sign[:,-1:].to(device)
        res = bool_sign[:,-1:].clone()
        for i in range(m-2,-1,-1):
            res += bool_sign[:,i:(i+1)].to(device)
            results = torch.cat((torch.sign(res.clone()),results),1).to(device)


        return results
