import pickle
import numpy as np
from scipy.stats import binom
import scipy
import math
from scipy.optimize import fsolve
import random

class risk_control:

    def calculate_bound(self,delta,m,erm):
        #This function is a solver for the inverse of binomial CDF using binary search.
        persicion = 1e-7
        def func(b):
            return (-1*delta) + scipy.stats.binom.cdf(int(m*erm),m,b)
        a=erm #start binary search from the empirical risk
        c=1   # the upper bound is 1
        b = (a+c)/2 #mid point
        funcval  =func(b)
        while abs(funcval)>persicion:
            if funcval>0:
                a=b
            else:
                c=b
            b = (a + c) / 2
            funcval = func(b)
        return b

    def bound(self,rstar,delta,kappa,residuals,split=True):
        # A function to calculate the bound
        #Input: rstar - the requested risk bound
        #       delta - the desired delta
        #       kappa - rating function over the points (higher values is more confident prediction)
        #       residuals - a vector of the residuals of the samples 0 is correct prediction and 1 corresponding to an error
        #       split - is a boolean controls whether to split train and test
        #Output - void (prints latex text for tables in the paper)

        # when spliting to train and test this represents the fraction of the validation size
        valsize = 0.5

        probs = kappa
        FY = residuals


        if split:
            idx = list(range(len(FY)))
            random.shuffle(idx)
            slice = round(len(FY)*(1-valsize))
            FY_val = FY[idx[slice:]]
            probs_val = probs[idx[slice:]]
            FY = FY[idx[:slice]]
            probs = probs[idx[:slice]]
        m = len(FY)

        probs_idx_sorted = np.argsort(probs)

        a=0
        b = m-1
        deltahat = delta/math.ceil(math.log2(m))

        for q in range(math.ceil(math.log2(m))+1):
            mid = math.ceil((a+b)/2)

            mi = len(FY[probs_idx_sorted[mid:]])
            theta = probs[probs_idx_sorted[mid]]
            risk = sum(FY[probs_idx_sorted[mid:]])/mi
            if split:
                testrisk = sum(FY_val[probs_val>=theta])/len(FY_val[probs_val>=theta])
                testcov = len(FY_val[probs_val>=theta])/len(FY_val)
            bound = self.calculate_bound(deltahat,mi,risk)
            coverage = mi/m
            if bound>rstar:
                a=mid
            else:
                b=mid
        print(b-mid)

        if split:
            print("%.2f & %.4f & %.4f & %.4f & %.4f & %.4f  \\\\" % (rstar,risk,coverage,testrisk,testcov,bound))
        else:
            print("%.2f & %.4f & %.4f & %.4f   \\\\" % (rstar,risk,coverage,bound))




