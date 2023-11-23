import os
import sys
sys.path.append(os.path.abspath(r'/Users/fuchuenli/Desktop/Data Science /Year 2/Trimester 2/COMP SCI 7097/GP-SHAP/Shapley Prior'))
sys.path.append(os.path.abspath(r'/Users/fuchuenli/Desktop/Data Science /Year 2/Trimester 2/COMP SCI 7097/RKHS-SHAP'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import pandas as pd
import seaborn as sns

from gpytorch.kernels import RBFKernel
from sklearn.datasets import load_diabetes
import matplotlib.pylab as plt
from gpytorch.lazy import lazify

from src.gp_model.VariationalGPRegression import VariationalGPRegression
from src.explanation_algorithms.BayesGPSHAP import BayesGPSHAP
from src.predictive_explanation.ShapleyPrior import ShapleyPrior
from src.utils.visualisation import summary_plot
from src.predictive_explanation.ShapleyKernel import ShapleyKernel
from sklearn.datasets import fetch_california_housing
from math import comb

import fastshap
import time

import os, sys, pickle, warnings, torch, shap

import numpy as np
from experiments.BananaShapley.banana_distribution import Banana2d
from experiments.BananaShapley.gshap_banana import Observation2dBanana

warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error, pairwise_distances, r2_score
import copy

import torch
import torch.nn as nn
from fastshap.utils import MaskLayer1d
from fastshap import Surrogate, KLDivLoss
from fastshap import FastSHAP

if __name__ == "__main__":
    n = 1000
    v = 10
    b = 0.01
    iterations = 10
    gshap_result = []
    fastshap_result = []
    shapley_prior_result = []
    for _ in range(10):
        banana2d = Banana2d(n=n, v=v, b=b, noise=0, outlier_quantile=2.)

        # Scaled the output so that can we compare accuracy across
        scale = 1
        y = torch.Tensor(banana2d.y/scale)
        X = torch.Tensor(banana2d.X)
        d = X.shape[1]

        compute_mh = lambda X: np.array([np.median(pairwise_distances(X[:, [i]])) for i in range(X.shape[1])])
        lengthscale = torch.tensor(compute_mh(X)).float()
        lengthscale[1] *= 1
        print("Lengthscale:", lengthscale)

        # True OSVs:
        phi1 = banana2d.phi_1/scale
        phi2 = banana2d.phi_2/scale
        PHI = np.array([phi1, phi2]).T

        cutting_points = 800

        X_train = X[:cutting_points]
        y_train = y[:cutting_points]
        X_test = X[cutting_points:]
        y_test = y[cutting_points:]

        PHI_train = PHI[:cutting_points]
        PHI_test = PHI[cutting_points:]

        kernel = RBFKernel
        gp_regression = VariationalGPRegression(
            X_train, y_train, kernel=kernel, num_inducing_points=200, batch_size=128)

        gp_regression.fit(learning_rate=1e-2,
                        training_iteration=300)

        surr = nn.Sequential(
            MaskLayer1d(value=0, append=True),
            nn.Linear(2 * d, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 1)
        )

        surrogate = Surrogate(surr, d)

        def original_model(x):
            return gp_regression.predict(x).mean.detach().reshape(-1, 1).float()

        surrogate.train_original_model(
            X_train,
            X_test,
            original_model,
            batch_size=64,
            max_epochs=100,
            loss_fn=nn.MSELoss(),
            validation_samples=10,
            validation_batch_size=10000,
            verbose=True
        )

        explainer = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2 * d))

        # Set up FastSHAP object
        fastshap = FastSHAP(explainer, surrogate, normalization='additive')

        # Train
        fastshap.train(
            X_train,
            X_test,
            batch_size=32,
            num_samples=32,
            max_epochs=1000,
            validation_samples=128,
            verbose=True)

        # fastshap predictions
        fastshap_preds = [fastshap.shap_values(X_test[i:i+1])[0].mean(axis=1) for i in range(X.shape[0] - cutting_points)]
        fastshap_preds = torch.tensor(fastshap_preds).t()

        fastshap_mse = mean_squared_error(y_true=PHI_test, y_pred=fastshap_preds.reshape((-1,2)))
        fastshap_result.append(fastshap_mse)

        bayesgpshap = BayesGPSHAP(train_X=X, kernel=RBFKernel(), gp_model=gp_regression,
                                include_likelihood_noise_for_explanation=False, scale=scale)

        bayesgpshap.run_bayesSHAP(X=X, num_coalitions=2**d)
        explanations = bayesgpshap.mean_shapley_values

        target = explanations.t().reshape(-1, 1)
        shapley_kernel = ShapleyKernel(
            train_X=X, kernel=RBFKernel(), lengthscales=gp_regression.lengthscale, 
            inducing_points=gp_regression.inducing_points,
            num_coalitions=2**d, sampling_method="subsampling", verbose=False
        )

        target_train = target[:cutting_points*d]

        optim = torch.optim.Adam(shapley_kernel.parameters(), lr=1e-3)
        def loss_function(pred, true):
            return torch.mean((true - pred) **2)

        min_loss = np.inf
        early_stopping = 0
        while True:
            optim.zero_grad()
            Psi = shapley_kernel(X)
            K = torch.einsum("ijk,lmn->imkn", Psi, Psi.transpose(0, 1))
            K = K.permute(2, 0, 3, 1).resize(len(target), len(target))
            K_train = K[:cutting_points*d, :cutting_points*d]
            prediction = K_train @ lazify(K_train).add_diag(shapley_kernel.krr_regularisation).inv_matmul(target_train)
            loss = loss_function(prediction, target_train)
            print(loss)
            loss.backward()
            optim.step()
            if loss >= min_loss:
                early_stopping += 1
            else:
                min_loss = loss
                early_stopping = 0
                
            if early_stopping >= 5:
                break

        Psi = shapley_kernel(X)
        K = torch.einsum("ijk,lmn->imkn", Psi, Psi.transpose(0, 1))
        K = K.permute(2, 0, 3, 1).resize(len(target), len(target))

        K_train = K[:cutting_points*d, :cutting_points*d]
        K_test_train = K[cutting_points*d:, :cutting_points*d]

        train_target = target[:cutting_points*d]
        test_target = target[cutting_points*d:]

        pred = K_test_train @ lazify(K_train).add_diag(shapley_kernel.krr_regularisation).inv_matmul(train_target)
        pred = pred.reshape((-1, 2)).detach().numpy()

        shapley_prior_mse = mean_squared_error(y_true=PHI_test, y_pred=pred)
        shapley_prior_result.append(shapley_prior_mse)

        # Assume this is your custom model class
        class CustomModel:
            def predict_custom(self, data):
                # Your prediction logic here
                pred = []
                for i in range(data.shape[0]):
                    pred.append(gp_regression.predict(torch.Tensor(data[i]).reshape((1,d))).loc.item())
                return np.array(pred)

        # Create an instance of your model
        model = CustomModel()

        # Define a wrapper function for the model's prediction
        def model_predict(data):
            return model.predict_custom(data)

        ogshap = Observation2dBanana(model_predict, X_test.numpy())
        ophi1, ophi2 = ogshap.fit(X_test, num_samples=X_test.shape[0])    
        OPHI = np.array([ophi1,ophi2]).T

        gshap_mse = mean_squared_error(y_true=PHI_test, y_pred=OPHI)
        gshap_result.append(gshap_mse)

    results = pd.DataFrame({"SHAP": ["GSHAP", "FastSHAP", "ShapleyPrior"], "MSE": [np.mean(gshap_result), np.mean(fastshap_result), np.mean(shapley_prior_result)]})
    sns.barplot(x="SHAP", y="MSE", data=results)
    plt.title("SHAP Value Comparison")
    plt.xlabel("SHAP")
    plt.ylabel("MSE")
    plt.savefig("shap_value_comparison.png")