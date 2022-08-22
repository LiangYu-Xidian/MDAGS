import math
import random
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from chemprop.utils import load_args, load_checkpoint,load_scalers
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

class Optimization():
    def __init__(self):
        self.fit_max = 0.7
        self.fit_min = 0
        self.NP = 600
        self.FES = 120000
        self.e = 0.0001
        self.w = 0.8
        self.r1 = 0.5
        self.r2 = 0.5
        self.dim = 1900
        self.theta=0.5
        self.degree=0.001
        self.v = np.zeros((self.NP, self.dim))
        self.x = np.zeros((self.NP, self.dim))
        self.fit = np.zeros((self.NP))
        self.gbest = -1
        self.fgbest = 10
        self.scalers = load_scalers(checkpoint_paths)
        self.cov = list(np.eye(self.dim) * 0.00002)
        self.mean = np.zeros((self.dim))

    def f(self, x):
        x=[x]
        y_pred=gpr.predict(x, return_std=False)
        y_pred=y_pred[0]
        return y_pred

    def run(self,candidate):
        best_fingerprint = []
        best_fit = []
        perturbation = self.theta * np.random.multivariate_normal(mean=self.mean, cov=self.cov, size=self.NP)
        for i in range(self.NP):
            for j in range(self.dim):
                self.v[i, j] = 0.001 * random.uniform(0,1)
            if i==0:
                self.x[i]=candidate
                self.fit[i] = self.f(self.x[i])
                self.fit_max = self.fit[i]
            else:
                self.x[i]=copy.deepcopy([i + j for i, j in zip(self.x[0],perturbation[i])])
                self.fit[i] = self.f(self.x[i])

            if self.fit[i] >= self.fit_min and self.fit[i] <= self.fit_max:
                best_fit.append(self.fit[i])
                best_fingerprint.append(copy.deepcopy(self.x[i]))
        pfgbest = min(self.fit)
        if pfgbest < self.fgbest:
            self.gbest = list(self.fit).index(pfgbest)
            self.fgbest = pfgbest
        fes = self.NP
        while fes < self.FES:
            for i in range(self.NP):
                b1=i
                b2=i
                while b1==i:
                    b1 = random.randint(0, self.NP-1)
                while b2==i and b2==b1:
                    b2 = random.randint(0, self.NP-1)
                if self.fit[b1] <= self.fit[i] and self.fit[b2] <= self.fit[i]:
                    if self.fit[b2] < self.fit[b1]:
                        temp=b1
                        b1=b2
                        b2=temp
                    beta = 0.5 - 0.2 * math.exp(-((self.fit[i] - self.fit[b2] + self.e)/(self.fit[i] - self.fit[b1] + self.e)))
                    self.v[i] = self.w * self.v[i] + self.r1 * (self.x[b1] - self.x[i]) + beta * self.r2 * (self.x[b2] - self.x[i])
                    self.x[i] = self.x[i]+self.v[i]
                    self.fit[i] = self.f(self.x[i])
                    if self.fit[i]>=self.fit_min  and self.fit[i]<=self.fit_max:
                        best_fit.append(self.fit[i])
                        best_fingerprint.append(copy.deepcopy(self.x[i]))
                    fes += 1
            pfgbest = min(self.fit)
            if pfgbest < self.fgbest:
                self.gbest = list(self.fit).index(pfgbest)
                self.fgbest = pfgbest

        return best_fingerprint,best_fit


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    checkpoint_paths = 'chemprop\\trained\\fold_0\\model_0\\model.pt'
    row_path = 'chemprop\\data\\train_regression.csv'
    filename = 'chemprop\\data\\regression_train_fingerprint.csv'
    data = pd.read_csv(filename, sep=',')
    X = np.array(np.array(data.iloc[:, 1:-1]))
    # print('X',X)
    y = np.array(np.array(data.iloc[:, -1]))
    # y = [float(i) for i in y]
    # print('y',y)
    print('start')
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)

    candidate = []
    filename = 'chemprop\\data\\seed.csv'
    data = pd.read_csv(filename, sep=',')
    candidate = np.array(np.array(data.iloc[:, 1:-1]))
    # self.NP=len(self.x)
    s_best_fingerprint=[]
    s_best_fit=[]
    num_seed = candidate.shape[0]
    for i in range(num_seed):
        best_fingerprint,best_fit=Optimization().run(candidate[i])
        s_best_fingerprint.extend(copy.deepcopy(best_fingerprint[1:]))
        s_best_fit.extend(copy.deepcopy(best_fit[1:]))
        #print(s_best_fingerprint)
    s_best_fit=pd.DataFrame(s_best_fit)
    s_best_fingerprint=pd.DataFrame(s_best_fingerprint)


    s_best_fingerprint.to_csv('chemprop\\preds_path\\seed\\seed_fingerprint_GP.csv', index=False)
    s_best_fit.to_csv('chemprop\\preds_path\\seed\\seed_fit_GP.csv', index=False)
