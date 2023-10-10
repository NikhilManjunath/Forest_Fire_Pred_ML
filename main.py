# Importing the required Libraries
from typing import Any
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
# from keras.models import Sequential
# from keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Importing our custom functions
import utils
import confusion_matrix

#Suppress warnings from Sklearn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Class to load our data
class Dataloader:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def load_data(self):
        data = pd.read_csv(self.data_path)
        return data

# Preprocessing Pipeline
class DataPreprocessor:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def preprocess_data(self):
        self._convert_date_to_day_of_week()
        self._add_weekday_weekend_column()
        self._expand_features()
        self._standardize_data()
        X_train, X_test, y_train, y_test = self._split_and_get_labels()
        return X_train, X_test, y_train, y_test

    # Get Day from Date
    def _convert_date_to_day_of_week(self):
        self.train['Date'] = pd.to_datetime(self.train['Date'], dayfirst=True)
        self.test['Date'] = pd.to_datetime(self.test['Date'], dayfirst=True)

    # Create a new feature - Weekday/Weekend to check if probability of forest fire is greater amongst either
    def _add_weekday_weekend_column(self):
        self.train['Weekday/Weekend'] = (self.train['Date'].dt.day_of_week < 5).astype(int)
        self.test['Weekday/Weekend'] = (self.test['Date'].dt.day_of_week < 5).astype(int)

    # Polynomial Expansion to create new features from existing features
    def _expand_features(self):
        columns = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']
        poly = PolynomialFeatures(2)
        self.X_train_expanded = poly.fit_transform(self.train[columns])
        self.X_test_expanded = poly.transform(self.test[columns])
        feature_names = poly.get_feature_names_out(input_features=None)
        self.X_train_expanded = pd.DataFrame(self.X_train_expanded, columns=feature_names)
        self.X_test_expanded = pd.DataFrame(self.X_test_expanded, columns=feature_names)
        self.X_train_expanded = self.X_train_expanded.drop(columns=['1'])
        self.X_test_expanded = self.X_test_expanded.drop(columns=['1'])

        # Reset Index
        self.train.reset_index()
        self.test.reset_index()


    # Standardize the dataset
    def _standardize_data(self):
        feature_columns = [column for column in self.X_train_expanded.columns]
        scaler = StandardScaler()
        self.X_train_expanded = pd.DataFrame(scaler.fit_transform(self.X_train_expanded),columns=feature_columns)
        self.X_test_expanded = pd.DataFrame(scaler.transform(self.X_test_expanded), columns=feature_columns)

    def _split_and_get_labels(self):
        self.X_train_expanded.insert(0, "Weekday/Weekend", train['Weekday/Weekend'])
        self.X_test_expanded.insert(0, "Weekday/Weekend", test['Weekday/Weekend'])
        self.y_train_expanded = self.train['Classes']
        self.y_test_expanded = self.test['Classes']
        return self.X_train_expanded, self.X_test_expanded, self.y_train_expanded, self.y_test_expanded

# Machine Learning Models
class ModelTrainer:
    def __init__(self,model_name):
        self.model_name = model_name
        self.models = {
            'SVM':
            {
                'model': SVC(),
                'param_grid': 
                {
                    'C': [0.1, 1, 10, 100, 1000],
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'kernel': ['rbf', 'linear', 'poly']
                }
            },

            'KNN':
            {
                'model': KNeighborsClassifier(),
                'param_grid': 
                {
                    'n_neighbors': [1,2,3,4,5,6,7,8,9,10],
                    'weights': ['uniform','distance'],
                    'metric': ['minkowski','chebyshev','euclidean']
                }
            },

            'Ridge':
            {
                'model': RidgeClassifier(),
                'param_grid': 
                {
                    'alpha': [0.1,1,10],
                    'solver': ['auto','svd','cholesky','lsqr','sparse_cg','sag','saga']
                }
            },

            'Logistic_Regression':
            {
                'model': LogisticRegression(),
                'param_grid': 
                {
                    'solver': ['newton-cg','liblinear','lbfgs'],
                    'penalty': ['l2']
                }
            },

            'Random_Forest':
            {
                'model': RandomForestClassifier(),
                'param_grid': 
                {
                    'max_features': ['auto','sqrt','log2'],
                    'criterion': ['gini','entropy'],
                    'n_estimators': [100,200,300,350]
                }
            },
        }

    # Training and Cross Validation Function
    def train_and_cross_validation(self, X_train, y_train):
        if self.model_name not in self.models:
            raise ValueError('Model {0} is not supported'.format(self.model_name))
        
        model_info = self.models[self.model_name]
        model = model_info['model']
        param_grid = model_info['param_grid']

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=5)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print('Best Hyperparameters for {0}: {1}'.format(self.model_name, best_params))
        print('Best Cross-Validation F1 Score: {0}'.format(np.round(best_score,2)))

        return model, grid_search.best_score_, grid_search.best_params_

# Experiments
class Experiments:
    def __init__(self):
        self.model_list = ['SVM', 'KNN', 'Ridge', 'Logistic_Regression','Random_Forest']
        self.best_model = ''
        self.best_score = 0
        self.best_params = []
        self.best_data = ''

    def data_experiments(self, X_train, y_train, X_test):
        #PCA Features
        X_train_PCA, X_test_PCA = utils.PCA_func(X_train, X_test,9)
        #SFS Features
        sfs_columns = utils.seq_feat_selector(X_train,y_train,9)
        X_train_SFS = X_train[sfs_columns]

        data_dict = {
            'PCA': X_train_PCA,
            'SFS': X_train_SFS
        }

        for data in data_dict:
            print('Training on {0} features'.format(data))
            for model in self.model_list:
                print('{0}:'.format(model))
                trainer = ModelTrainer(model)
                model_fn,score,params = trainer.train_and_cross_validation(data_dict[data], y_train)

                if score > self.best_score:
                    self.best_model = model_fn
                    self.best_score = score
                    self.best_params = params
                    self.best_data = data
                print()
        
        if self.best_data == 'PCA':
            return self.best_model, self.best_params, self.best_score, X_train_PCA, X_test_PCA
        else:
            return self.best_data, self.best_model, self.best_params, self.best_score, X_train[sfs_columns], X_test[sfs_columns]


if __name__ == '__main__':

    # Loading Train and Test data
    train_loader = Dataloader('./data/algerian_fires_train.csv')
    test_loader = Dataloader('./data/algerian_fires_test.csv')
    train = train_loader.load_data()
    test = test_loader.load_data()

    # Preprocessing
    preprocessor = DataPreprocessor(train,test)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data()
    utils.PCA_analysis(X_train)

    # Model Selection and Hyperparameter Tuning
    exp = Experiments()
    fs_type, model, params, score, X_train_fs, X_test_fs = exp.data_experiments(X_train,y_train,X_test)
    
    # Testing on Test Data
    print('Testing Models on Test Data')    
    # 1. Trivial Model - Random Classifier
    print('Trivial Model')
    trivial_metrics = utils.trivial(X_train, y_train, X_test, y_test)
    print('Performance of Trivial Model on Test Data: Accuracy: {0}, F1 Score: {1}'.format(trivial_metrics['accuracy'], trivial_metrics['f1_score']))
    print()

    # 2. Baseline Model - Nearest Means Classifier
    print('Baseline Nearest Means Model')
    baseline_metrics = utils.nearest_means(train,test)
    print('Performance of Baseline Nearest Means Model on Test Data: Accuracy: {0}, F1 Score: {1}'.format(trivial_metrics['accuracy'], trivial_metrics['f1_score']))
    print()

    # 3. Best ML Model - KNN over SFS Data
    model.set_params(**params)
    model.fit(X_train_fs, y_train)
    y_test_pred = model.predict(X_test_fs)
    metrics = utils.compute_metrics(y_test_pred,y_test.to_numpy())
    print(classification_report(y_test.to_numpy(),y_test_pred))
    acc_score=accuracy_score(y_test,y_test_pred)
    f1score = f1_score(y_test,y_test_pred)
    sns.heatmap(confusion_matrix.calc_confusion_matrix(y_test,y_test_pred),annot=True)
    print('Performance of Best ML Model on Test Data: Accuracy: {0}, F1 Score: {1}'.format(np.round(acc_score,2), np.round(f1score,2)))

    










    

        