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
        self.NP = 600
        self.FES = 120000
        self.e = 0.0001
        self.w = 0.8
        self.r1 = 0.5
        self.r2 = 0.5
        self.dim = 1900
        self.theta=0.2
        self.v = np.zeros((self.NP, self.dim))
        self.x = np.zeros((self.NP, self.dim))
        self.fit = np.zeros((self.NP, 1))
        self.gbest = -1
        self.fgbest = 10
        self.fit_max = 0.8
        self.fit_min = -0.3

        self.model = load_checkpoint(checkpoint_paths)
        self.model.eval()
        self.scalers = load_scalers(checkpoint_paths)
        self.cov = list(np.eye(self.dim) * 0.00002)
        self.mean = np.zeros((self.dim))


    def f(self, x):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(x).to(torch.float32)
            pred=self.model.ffn(x).detach().numpy()
            scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler = self.scalers
            pred=scaler.inverse_transform(pred)
        return pred

    def run(self, candidate):
        best_fingerprint = []
        best_fit = []
        perturbation = self.theta * np.random.multivariate_normal(mean=self.mean, cov=self.cov, size=self.NP)
        for i in range(self.NP):
            for j in range(self.dim):
                self.v[i, j] = random.uniform(0, 0.00001)
            if i == 0:
                self.x[i] = candidate
                self.fit[i] = self.f(self.x[i])
                self.fit_max = self.fit[i]
                print('pred',self.fit[i])
            else:
                # gauss=[self.theta*self.degree*(random.gauss(0, 1)) for i in range(self.dim)]
                self.x[i] = copy.deepcopy([i + j for i, j in zip(self.x[0], perturbation[i])])
                self.fit[i] = self.f(self.x[i])

            if self.fit[i] >= self.fit_min and self.fit[i] <= self.fit_max:
                best_fit.append(self.fit[i][0])
                best_fingerprint.append(copy.deepcopy(self.x[i]))

        fes = self.NP
        while fes < self.FES:
            for i in range(self.NP):
                b1 = i
                b2 = i
                while b1 == i:
                    b1 = random.randint(0, self.NP - 1)
                while b2 == i or b2 == b1:
                    b2 = random.randint(0, self.NP - 1)
                if self.fit[b1] <= self.fit[i] and self.fit[b2] <= self.fit[i]:
                    if self.fit[b2] < self.fit[b1]:
                        temp = b1
                        b1 = b2
                        b2 = temp
                    beta = 0.5 - 0.2 * math.exp(
                        -((self.fit[i] - self.fit[b2] + self.e) / (self.fit[i] - self.fit[b1] + self.e)))
                    self.v[i] = self.w * self.v[i] + self.r1 * (self.x[b1] - self.x[i]) + beta * self.r2 * (
                                self.x[b2] - self.x[i])
                    self.x[i] = self.x[i] + self.v[i]
                    self.fit[i] = self.f(self.x[i])
                    if self.fit[i] >= self.fit_min and self.fit[i] <= self.fit_max:
                        best_fit.append(self.fit[i][0])
                        best_fingerprint.append(copy.deepcopy(self.x[i]))
                    fes += 1
        pfgbest = min(self.fit)
        if pfgbest < self.fgbest:
            self.gbest = list(self.fit).index(pfgbest)
            self.fgbest = pfgbest

        return best_fingerprint, best_fit


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    checkpoint_paths = 'chemprop\\trained\\fold_0\\model_0\\model.pt'
    row_path = 'chemprop\\data\\train_regression.csv'
    filename = 'chemprop\\data\\seed.csv'
    data = pd.read_csv(filename, sep=',')
    candidate = np.array(np.array(data.iloc[:, 1:-1]))
    # self.NP=len(self.x)
    s_best_fingerprint = []
    s_best_fit = []
    num_seed = candidate.shape[0]
    Optimization=Optimization()
    for i in range(num_seed):
        best_fingerprint, best_fit = Optimization.run(candidate[i])
        print(i)
        print('len_best_fingerprint', len(best_fingerprint[1:]))
        print('len_best_fit', len(best_fit[1:]))
        s_best_fingerprint.extend(copy.deepcopy(best_fingerprint[1:]))
        s_best_fit.extend(copy.deepcopy(best_fit[1:]))
    s_best_fit = pd.DataFrame(s_best_fit)
    s_best_fingerprint = pd.DataFrame(s_best_fingerprint)
    print('len_fit', len(s_best_fit))
    print('len_fing', len(s_best_fingerprint))
    s_best_fingerprint.to_csv('preds_path\\seed\\seed_fingerprint_ffn.csv', index=False)
    s_best_fit.to_csv('preds_path\\seed\\seed_fit_ffn.csv', index=False)
