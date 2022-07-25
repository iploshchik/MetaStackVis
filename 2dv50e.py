
import streamlit as st

import pandas as pd
import time
import umap
from sklearn import preprocessing
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, f1_score, log_loss
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import os
from PIL import Image
import math
import hdbscan
from sklearn.preprocessing import minmax_scale
import shutil
from umap_functions import *
from coverage_function import *
from supporting_functions import *
from plotting_comparison import *

# define random seed
np.random.seed(42)

# Page layout
st.set_page_config(page_title='Visually-Assisted Performance Evaluation of Metamodels in Stacking Ensemble Learning',layout='wide')
st.subheader('Visually-Assisted Performance Evaluation of Metamodels in Stacking Ensemble Learning')


# Sidebar - Collects user input features into dataframes
# st.sidebar.subheader('Upload your original train dataset')
uploaded_source_train = st.sidebar.file_uploader("Upload your original train dataset", type=["csv"], key='source_train')
# st.sidebar.subheader('Upload your original test dataset')
uploaded_source_test = st.sidebar.file_uploader("Upload your original test dataset", type=["csv"], key='source_test')
# st.sidebar.subheader('Upload your target train data')
uploaded_target_train = st.sidebar.file_uploader("Upload your target_train data", type=["csv"], key='target_train')
# st.sidebar.subheader('Upload your target test data')
uploaded_target_test = st.sidebar.file_uploader("Upload your target_test data", type=["csv"], key='target_test')
# st.sidebar.subheader('Upload your model probabilities data')
uploaded_probabilities = st.sidebar.file_uploader("Upload your model probabilities data", type=["csv"], key='probabilities')
# st.sidebar.subheader('Upload your model data')
uploaded_model = st.sidebar.file_uploader("Upload your model data", type=["csv"], key='model')


# Sidebar - Specify parameter settings
st.sidebar.subheader('Set Parameters for HDBSCAN plot')
parameter_hdb_cluster_size = st.sidebar.number_input('Min Cluster Size', 3, key='hdb_min_cluster_size')
parameter_hdb_min_samples = st.sidebar.number_input('Min number of Samples', 5, key='hdb_min_samples')
parameter_hdb_metrics = st.sidebar.selectbox('Metric', ['euclidean', 'manhattan', 'chebyshev'], key='hdb_metric')
st.sidebar.write('---')
st.sidebar.subheader('Set Parameters for UMAP Chart')
parameter_umap_n_neighbors = st.sidebar.number_input('Number of neighbors (n_neighbors)', 15, key='n_neighbors')
parameter_umap_metric = st.sidebar.selectbox('Metric', ('euclidean', 'manhattan', 'chebyshev', 'minkowski'), key='metric')
parameter_umap_min_dist = st.sidebar.number_input('Minimal distance', 0.2, key='min_dist')
st.sidebar.write('---')
st.sidebar.subheader('General Plotly charts Parameters')
plotly_size = st.sidebar.number_input('Size', 600, key='size')
st.sidebar.write('---')


#------------------------------------------------------------------------------------------------


if uploaded_source_train is not None and uploaded_source_test is not None and uploaded_target_train is not None and             uploaded_target_test is not None and uploaded_probabilities is not None and uploaded_model is not None:

        # Read in data
        df_source_train = pd.read_csv(uploaded_source_train)
        df_source_test = pd.read_csv(uploaded_source_test)
        df_prob = pd.read_csv(uploaded_probabilities)
        df_mod = pd.read_csv(uploaded_model)

        df_target_train = pd.read_csv(uploaded_target_train)
        y_train = df_target_train.copy()
        df_target_test = pd.read_csv(uploaded_target_test)
        y_test = df_target_test.copy()

        scaler = preprocessing.StandardScaler()

        df_source_train_scaled = pd.DataFrame(scaler.fit_transform(df_source_train), columns=df_source_train.columns)
        x_train = df_source_train_scaled.copy()
        df_source_test_scaled = pd.DataFrame(scaler.fit_transform(df_source_test), columns=df_source_test.columns)
        x_test = df_source_test_scaled.copy()

        # Check for missing values and print values per column if any are missing for each of dataframes
        if df_prob.isnull().values.any() or df_mod.isnull().values.any() or df_source_train.isnull().values.any() or df_source_test.isnull().values.any() or df_target_train.isnull().values.any() or df_target_test.isnull().values.any():
                st.write('There are some missing values in some of the dataframes, please correct datasets and re-upload the files')
                st.write(f'The number of missing values in model probabilities dataset is {df_prob.isnull().sum()}')
                st.write(f'The number of missing values in top models dataset is {df_mod.isnull().sum()}')
                st.write(f'The number of missing values in source train dataset is {df_source_train.isnull().sum()}')
                st.write(f'The number of missing values in source test dataset is {df_source_test.isnull().sum()}')
                st.write(f'The number of missing values in target dataset is {df_target_train.isnull().sum()}')
                st.write(f'The number of missing values in original dataset is {df_target_test.isnull().sum()}')


        algo_nr = df_mod['algorithm_id']

        algos = {1:'K-Nearest Neighbor', 2:'Support Vector Machine', 3:'Gaussian Naive Bayes', 4:'Multilayer Perceptron', 5:'Logistic Regression',
        6:'Linear Discriminant Analysis', 7:'Quadratic Discriminant Analysis', 8:'Random Forest', 9:'Extra Trees', 10:'Adaptive Boosting',
        11:'Gradient Boosting'}

        symbols = ['circle', 'square', 'x', 'cross', 'diamond', 'star', 'hexagram', 'triangle-right', 'triangle-left', 'triangle-down', 'triangle-up']


        df_model = df_mod.copy()
        # rename columns in df_model
        df_model.rename(columns={'params': 'hyperparameters', 'mean_test_accuracy': 'accuracy', 'mean_test_precision_weighted': 'precision', 'mean_test_recall_weighted': 'recall', 'mean_test_roc_auc_ovo_weighted': 'roc_auc_score',
        'geometric_mean_score_weighted': 'geometric_mean_score'}, inplace=True)
        # remove overall perfromance column
        df_model.drop(columns=['overall_performance'], inplace=True)
        df_model['algorithm_name'] = df_model['algorithm_id'].map(algos)
        df_model['overall_performance'] = round((df_model['accuracy'] + df_model['precision'] + df_model['recall'] + df_model['roc_auc_score'] + df_model['geometric_mean_score'] + df_model['matthews_corrcoef'] + df_model['f1_weighted']) / 7, 2)
        # Sort columns
        df_model = df_model[['model_id', 'algorithm_id', 'algorithm_name', 'accuracy', 'precision', 'recall', 'roc_auc_score',
        'geometric_mean_score', 'matthews_corrcoef', 'f1_weighted', 'log_loss', 'overall_performance', 'hyperparameters']]


        # create new pandas series with total number of models
        prob_base = []

        for i in range(df_prob.shape[0]):
                prob = []
                # calculate the confidence interval for each algorithm
                for n in range(df_prob.shape[1]):
                        prob.append(df_prob.iloc[i, n])
                prob_average = np.mean(prob).round(2)
                prob_base.append(prob_average)
                # add prob_meta to df_model_meta as new column "average_probability"
                df_model['average_probability'] = pd.Series(prob_base)

        df_model['rank'] = df_model['overall_performance'] * df_model['average_probability'] / 10000


        # Metamodel estimators (top 1 per algorithm)

        # Hyperparameters will be extracted from the best performing model per algorithm. <br> The plan is to extract hyperparameters from the best performing model per algorithm and then apply these hyperparameters to the metamodels. Currently no functuonality to be added to allow end user to tune the hyperparameters for metamodels, instead they will rely on top performing ones from base layer models.

        # Extract the top models from the dataframe with the respective hyperparameters
        meta_params = best_params(df_model)

        # return unique values of algorithm name
        algo = ['knn', 'svm', 'gnb', 'mlp', 'lr', 'lda', 'qda', 'rf', 'et', 'ab', 'gb']
        # convert all values in algo to capital letters
        algo_cap = [i.upper() for i in algo]

        algo_names = meta_params['algorithm_name'].copy()

        # iterate through key and value pairs in algo
        i = 0
        for k,v in zip(algo, algo_names):
                # Define hyperparameters for each  final estimator, based on top performing model from base layer
                temp = get_hyperparameters(meta_params, v)
                # convert to dictionary for easy access
                temp_dict = {int(k):v for k,v in temp.items()}
                # return first value
                temp_dict = temp_dict[i]
                # convert to dictionary for easy access, assign varibale name per algorithm
                globals()[k + '_best_params'] = ast.literal_eval(temp_dict)
                i += 1

        # Add final estimators
        final_estimators = [
                ('knn', KNeighborsClassifier(algorithm=get_value(knn_best_params, 'algorithm'), metric=get_value(knn_best_params, 'metric'), n_neighbors= get_value(knn_best_params, 'n_neighbors'), weights=get_value(knn_best_params, 'weights'), n_jobs=-1)),
                ('svm', SVC(C=get_value(svm_best_params, 'C'), kernel=get_value(svm_best_params, 'kernel'),probability=True, random_state=42)),
                ('gnb', GaussianNB(var_smoothing=get_value(gnb_best_params, 'var_smoothing'))),
                ('mlp', MLPClassifier(activation=get_value(mlp_best_params, 'activation'), alpha=get_value(mlp_best_params, 'alpha'), max_iter=get_value(mlp_best_params, 'max_iter'), solver=get_value(mlp_best_params, 'solver'), tol=get_value(mlp_best_params, 'tol'), random_state=42)),
                ('lr', LogisticRegression(C=get_value(lr_best_params, 'C'), max_iter=get_value(lr_best_params, 'max_iter'),
                penalty=get_value(lr_best_params, 'penalty'), solver=get_value(lr_best_params, 'solver'), random_state=42, n_jobs=-1)),
                ('lda', LinearDiscriminantAnalysis(shrinkage=get_value(lda_best_params, 'shrinkage'), solver=get_value(lda_best_params, 'solver'))),
                ('qda', QuadraticDiscriminantAnalysis(reg_param=get_value(qda_best_params, 'reg_param'), tol=get_value(qda_best_params, 'tol'))),
                ('rf', RandomForestClassifier(criterion=get_value(rf_best_params, 'criterion'), n_estimators=get_value(rf_best_params, 'n_estimators'), random_state=42, n_jobs=-1)),
                ('et', ExtraTreesClassifier(criterion=get_value(et_best_params, 'criterion'), n_estimators=get_value(et_best_params, 'n_estimators'), random_state=42, n_jobs=-1)),
                ('ab', AdaBoostClassifier(algorithm=get_value(ab_best_params, 'algorithm'), learning_rate=get_value(ab_best_params, 'learning_rate'), n_estimators=get_value(ab_best_params, 'n_estimators'), random_state=42)),
                ('gb', GradientBoostingClassifier(learning_rate=get_value(gb_best_params, 'learning_rate'), n_estimators=get_value(gb_best_params, 'n_estimators'), random_state=42))
                ]


        # HDBSCAN

        # create new dataframe with with columns min_cluster_size, min_samples and number_outliers
        df_cluster = pd.DataFrame(columns=['min_cluster_size', 'min_samples', 'metric', 'n_clusters', 'DBVC', 'Coverage'])

        row = 0

        # Apply hdbscan to df_prob
        for i in [3,4,5,6, 7, 8]:
                for j in [5, 10, 15, 20, 25, 30, 40, 50]:
                        for metric in ['euclidean', 'manhattan', 'chebyshev']:
                                clusterer = hdbscan.HDBSCAN(min_cluster_size=i, min_samples=j, metric=metric, gen_min_span_tree=True)
                                clusterer.fit(df_prob)
                                labels = clusterer.labels_
                                cnts = pd.DataFrame(labels)[0].value_counts()
                                cnts = cnts.reset_index()
                                cnts.columns = ['cluster','count']
                                n_cluster = cnts.cluster.nunique()
                                # get DBVC
                                DBVC = clusterer.relative_validity_
                                clustered = (labels >= 0)
                                coverage = np.sum(clustered) / df_prob.shape[0]
                                # add values to dataframe
                                df_cluster.loc[row] = [i, j, metric, n_cluster, DBVC, coverage]
                                row += 1

        # add column with multiplication of DBVC and coverage
        df_cluster['DBVC_Coverage'] = df_cluster['DBVC'] * df_cluster['Coverage']
        # sort dataframe by DBVC_coverage
        df_cluster = df_cluster.sort_values('DBVC_Coverage', ascending=False)
        # group by DBVC_coverage
        df_cluster = df_cluster.groupby('DBVC_Coverage').apply(lambda x: x.sort_values('DBVC', ascending=False).iloc[0])
        # drop index and 
        df_cluster = df_cluster.reset_index(drop=True)

        # # keep only rows with number of clusters 4 or 5
        # df_cluster = df_cluster[df_cluster['n_clusters'].isin([4,5])]
        # sort by DBVC_coverage
        df_cluster = df_cluster.sort_values('DBVC_Coverage', ascending=False)


        with st.expander('Show clusters', expanded=True):
                st.subheader('top 5 cluster composition')
                st.table(df_cluster.head())

        
        st.write('Select HDBSCAN cluster settings from side menu or click to continue with settings according to highest score')

        if not st.button('Use settings for highest score'):
                # get min_cluster_size and min_samples from side menu
                min_cluster_size = st.session_state.hdb_min_cluster_size
                min_samples = st.session_state.hdb_min_samples
                # get metric from side menu
                metric = st.session_state.hdb_metric
        else:
                # get min_cluster_size and min_samples and from top row
                min_cluster_size = int(df_cluster.iloc[0]['min_cluster_size'])
                min_samples = int(df_cluster.iloc[0]['min_samples'])
                # get metric from top row
                metric = df_cluster.iloc[0]['metric']

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, gen_min_span_tree=True)
        clusterer.fit(df_prob)
        labels = clusterer.labels_

        cnts = pd.DataFrame(labels)[0].value_counts()
        cnts = cnts.reset_index()
        cnts.columns = ['cluster','count']
        # sort by cluster
        cnts = cnts.sort_values('cluster')

        # add labels to df_model
        df_model['labels'] = labels
        df_prob['labels'] = labels

        # show cents as table with title 'Clusters'
        st.subheader('Clusters')
        st.table(cnts)

        if st.checkbox('Calculate metamodel performance for each of the clusters and present HDBSCAN results'):

                # sort dataframes by labels
                df_model = df_model.sort_values('labels')
                df_prob = df_prob.reindex(df_model.index)

                # reset index for df_model and df_prob
                df_model = df_model.reset_index(drop=True)
                df_prob = df_prob.reset_index(drop=True)

                # create empty dictionaries to store results fro all clusters of basemodels and metamodels results
                df_model_dict = {}
                df_prob_dict = {}
                df_pred_dict = {}

                df_model_dict_meta = {}
                df_prob_dict_meta = {}
                df_pred_dict_meta = {}

                algo_dict = {}
                algo_names_dict = {}

                # create new dictionary with df_model name and content
                df_model_dict = {'df_model_all': df_model}
                df_model_dict_meta = {'df_model_all_meta': None}

                # save df_model subsets based on labels
                for label in df_model.labels.unique():
                        if label == -1:
                                # append to dictionary
                                df_model_dict['df_model_outliers'] = df_model[df_model['labels'] == -1]
                                df_model_dict_meta['df_model_outliers_meta'] = None
                        else:
                                # append to dictionary
                                df_model_dict[f'df_model_cluster_{label}'] = df_model[df_model['labels'] == label]
                                df_model_dict_meta[f'df_model_cluster_{label}_meta'] = None

                # create new dictionary with df_prob name and content
                df_prob_dict = {'df_prob_all': df_prob}
                df_prob_dict_meta = {'df_prob_all_meta': None}

                # save df_prob subsets based on labels
                for label in df_prob.labels.unique():
                        if label == -1:
                                # append to dictionary
                                df_prob_dict['df_prob_outliers'] = df_prob[df_prob['labels'] == -1]
                                df_prob_dict_meta['df_prob_outliers_meta'] = None
                        else:
                                # append to dictionary
                                df_prob_dict[f'df_prob_cluster_{label}'] = df_prob[df_prob['labels'] == label]
                                df_prob_dict_meta[f'df_prob_cluster_{label}_meta'] = None

                # drop label column from all dataframes in df_prob_dict
                for key in df_prob_dict.keys():
                        df_prob_dict[key] = df_prob_dict[key].drop(columns=['labels'])      

                # create new dictionary with predicitons
                df_pred_dict = {'df_pred_all': None}
                df_pred_dict_meta = {'df_pred_all_meta': None}
                # save df_prob subsets based on labels
                for label in df_prob.labels.unique():
                        if label == -1:
                                # append to dictionary
                                df_pred_dict['df_pred_outliers'] = None
                                df_pred_dict_meta['df_pred_outliers_meta'] = None
                        else:
                                # append to dictionary
                                df_pred_dict[f'df_pred_cluster_{label}'] = None
                                df_pred_dict_meta[f'df_pred_cluster_{label}_meta'] = None

                # create new dictionary with predicitons
                algo_dict = {'algo_all': None}
                # save df_prob subsets based on labels
                for label in df_prob.labels.unique():
                        if label == -1:
                                # append to dictionary
                                algo_dict['algo_outliers'] = None
                        else:
                                # append to dictionary
                                algo_dict[f'algo_cluster_{label}'] = None

                # create new dictionary with predicitons
                algo_names_dict = {'algo_all': None}
                # save df_prob subsets based on labels
                for label in df_prob.labels.unique():
                        if label == -1:
                                # append to dictionary
                                algo_names_dict['algo_outliers'] = None
                        else:
                                # append to dictionary
                                algo_names_dict[f'algo_cluster_{label}'] = None

                my_bar = st.progress(0)
                percent_complete = 0
                latest_iteration = st.text('')

                for key_model, key_prob, key_pred, key_algo, key_name_algo in zip(df_model_dict.keys(), df_prob_dict.keys(), df_pred_dict.keys(), algo_dict.keys(), algo_names_dict.keys()):

                        latest_iteration.text(f'Calculating metamodel performance for cluster {key_model}')

                        # dictionaries with algorithm names and their hyperparameters for all models
                        knn_params = {int(k):v for k,v in get_hyperparameters(df_model_dict[key_model], 'K-Nearest Neighbor').items()}
                        svm_params = {int(k):v for k,v in get_hyperparameters(df_model_dict[key_model], 'Support Vector Machine').items()}
                        gnb_params = {int(k):v for k,v in get_hyperparameters(df_model_dict[key_model], 'Gaussian Naive Bayes').items()}
                        mlp_params = {int(k):v for k,v in get_hyperparameters(df_model_dict[key_model], 'Multilayer Perceptron').items()}
                        lr_params = {int(k):v for k,v in get_hyperparameters(df_model_dict[key_model], 'Logistic Regression').items()}
                        lda_params = {int(k):v for k,v in get_hyperparameters(df_model_dict[key_model], 'Linear Discriminant Analysis').items()}
                        qda_params = {int(k):v for k,v in get_hyperparameters(df_model_dict[key_model], 'Quadratic Discriminant Analysis').items()}
                        rf_params = {int(k):v for k,v in get_hyperparameters(df_model_dict[key_model], 'Random Forest').items()}
                        et_params = {int(k):v for k,v in get_hyperparameters(df_model_dict[key_model], 'Extra Trees').items()}
                        ab_params = {int(k):v for k,v in get_hyperparameters(df_model_dict[key_model], 'Adaptive Boosting').items()}
                        gb_params = {int(k):v for k,v in get_hyperparameters(df_model_dict[key_model], 'Gradient Boosting').items()}

                        # Estimators and hyperparameters for each algorithm per df_model_cluster

                        estimators = []
                        # populate list of estimators with all 55 top models
                        count = 0
                        for i in list(knn_params.keys()):
                                        temp_dict = ast.literal_eval(knn_params[i])
                                        estimators.append((f'knn_{count}', KNeighborsClassifier(algorithm=get_value(temp_dict, 'algorithm'), 
                                                        metric=get_value(temp_dict, 'metric'), n_neighbors= get_value(temp_dict, 'n_neighbors'), 
                                                        weights=get_value(temp_dict, 'weights'), n_jobs=-1)))
                                        count += 1
                        count = 0
                        for i in list(svm_params.keys()):
                                        temp_dict = ast.literal_eval(svm_params[i])
                                        estimators.append((f'svm_{count}', SVC(C=get_value(temp_dict, 'C'), kernel=get_value(temp_dict, 'kernel'),probability=True, random_state=42)))
                                        count += 1
                        count = 0
                        for i in list(gnb_params.keys()):
                                        temp_dict = ast.literal_eval(gnb_params[i])
                                        estimators.append((f'gnb_{count}', GaussianNB(var_smoothing=get_value(temp_dict, 'var_smoothing'))))
                                        count += 1
                        count = 0
                        for i in list(mlp_params.keys()):
                                        temp_dict = ast.literal_eval(mlp_params[i])
                                        estimators.append((f'mlp_{count}', MLPClassifier(activation=get_value(temp_dict, 'activation'), 
                                                        alpha=get_value(temp_dict, 'alpha'),
                                                        max_iter=get_value(temp_dict, 'max_iter'), solver=get_value(temp_dict, 'solver'), 
                                                        tol=get_value(temp_dict, 'tol'), random_state=42)))
                                        count += 1
                        count = 0
                        for i in list(lr_params.keys()):
                                        temp_dict = ast.literal_eval(lr_params[i])
                                        estimators.append((f'lr_{count}', LogisticRegression(C=get_value(temp_dict, 'C'), max_iter=get_value(temp_dict, 'max_iter'), penalty=get_value(temp_dict, 'penalty'), solver=get_value(temp_dict, 'solver'), random_state=42, n_jobs=-1)))
                                        count += 1
                        count = 0
                        for i in list(lda_params.keys()):
                                        temp_dict = ast.literal_eval(lda_params[i])
                                        estimators.append((f'lda_{count}', LinearDiscriminantAnalysis(shrinkage=get_value(temp_dict, 'shrinkage'), 
                                                        solver=get_value(temp_dict, 'solver'))))
                                        count += 1
                        count = 0
                        for i in list(qda_params.keys()):
                                        temp_dict = ast.literal_eval(qda_params[i])
                                        estimators.append((f'qda_{count}', QuadraticDiscriminantAnalysis(reg_param=get_value(temp_dict, 'reg_param'), tol=get_value(temp_dict, 'tol'))))
                                        count += 1
                        count = 0
                        for i in list(rf_params.keys()):
                                        temp_dict = ast.literal_eval(rf_params[i])
                                        estimators.append((f'rf_{count}', RandomForestClassifier(criterion=get_value(temp_dict, 'criterion'), 
                                                        n_estimators=get_value(temp_dict, 'n_estimators'), random_state=42, n_jobs=-1)))
                                        count += 1
                        count = 0
                        for i in list(et_params.keys()):
                                        temp_dict = ast.literal_eval(et_params[i])
                                        estimators.append((f'et_{count}', ExtraTreesClassifier(criterion=get_value(temp_dict, 'criterion'),
                                                        n_estimators=get_value(temp_dict, 'n_estimators'), random_state=42, n_jobs=-1)))
                                        count += 1
                        count = 0
                        for i in list(ab_params.keys()):
                                        temp_dict = ast.literal_eval(ab_params[i])
                                        estimators.append((f'ab_{count}', AdaBoostClassifier(algorithm=get_value(temp_dict, 'algorithm'), 
                                                        learning_rate=get_value(temp_dict, 'learning_rate'), n_estimators=get_value(temp_dict, 'n_estimators'), random_state=42)))
                                        count += 1
                        count = 0
                        for i in list(gb_params.keys()):
                                        temp_dict = ast.literal_eval(gb_params[i])
                                        # update criterion as mae is deprecated
                                        if get_value(temp_dict, 'criterion') == 'mae':
                                                estimators.append((f'gb_{count}', GradientBoostingClassifier(criterion='squared_error', 
                                                        learning_rate=get_value(temp_dict, 'learning_rate'), n_estimators=get_value(temp_dict, 'n_estimators'), random_state=42)))
                                        else:
                                                estimators.append((f'gb_{count}', GradientBoostingClassifier(criterion=get_value(temp_dict, 'criterion'), 
                                                        learning_rate=get_value(temp_dict, 'learning_rate'), n_estimators=get_value(temp_dict, 'n_estimators'), random_state=42)))
                                        count += 1

                        
                        # Please note that for the average metric we did not include the logloss as using normalized version of logloss introduces bias to dataset

                        # create dataframe for df_model_meta with columns names from df_model
                        df_model_meta = pd.DataFrame(columns=df_model.columns)
                        # drop average probability and rank columns from df_model_meta
                        df_model_meta = df_model_meta.drop(['average_probability', 'rank'], axis=1)

                        # create dataframes for meta models probabilities and predicted values
                        df_pred_meta = pd.DataFrame()
                        df_prob_meta = pd.DataFrame()

                        for x in range(0, len(final_estimators)):
                                final_estimator = final_estimators[x][1]
                                clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator, n_jobs=-1, cv=5)
                                clf.fit(x_train, y_train)
                                y_pred = clf.predict(x_train)
                                y_pred = pd.Series(y_pred)

                                # probabilities dataframe
                                y_pred_prob = clf.predict_proba(x_train)
                                y_pred_prob_df = pd.DataFrame(y_pred_prob, columns=['prob_0', 'prob_1'])
                                y_pred_prob_df['target'] = y_train
                                y_pred_prob_df['predicted'] = y_pred
                                # create new column, set value to prob_0 if target is 0 and prob_1 if target is 1
                                y_pred_prob_df['pred_prob'] = np.where(y_pred_prob_df['target'] == 0, y_pred_prob_df['prob_0'], y_pred_prob_df['prob_1'])
                                # remove prob_0 and prob_1 columns, target and predicted columns
                                y_pred_prob_df = y_pred_prob_df.drop(['prob_0', 'prob_1', 'target', 'predicted'], axis=1)
                                # transpose the data frame and convert values to %
                                y_pred_prob_df = y_pred_prob_df.T
                                y_pred_prob_df = y_pred_prob_df.apply(lambda x: x * 100).round(2)
                                # set index to x
                                y_pred_prob_df.index = [x]
                                # add row to df_prob_meta using pd.concat
                                df_prob_meta = pd.concat([df_prob_meta, y_pred_prob_df], axis=0)
                                
                                # prediction dataframe
                                # transpose the data frame
                                y_pred_df = pd.DataFrame(y_pred, columns=['Pred_class'])
                                y_pred_df = y_pred_df.T
                                # set index to x
                                y_pred_df.index = [x]
                                # add row to df_prob_meta using pd.concat
                                df_pred_meta = pd.concat([df_pred_meta, y_pred_df], axis=0)

                                # performance metrics dataframe
                                accuracy = round(accuracy_score(y_train, y_pred)*100, 2)
                                precision = round(precision_score(y_train, y_pred, average='weighted')*100, 2)
                                recall = round(recall_score(y_train, y_pred, average='weighted')*100, 2)
                                roc_auc = round(roc_auc_score(y_train, y_pred, average='weighted')*100, 2)
                                gmean = round(geometric_mean_score(y_train, y_pred, average='weighted')*100, 2)
                                mcc = round(matthews_corrcoef(y_train, y_pred)*100, 2)
                                f1_weighted = round(f1_score(y_train, y_pred, average='weighted')*100, 2)
                                log_loss = round(metrics.log_loss(y_train, y_pred, normalize=True)*100, 2)
                                average_metrics = (accuracy + precision + recall + roc_auc + gmean + mcc + f1_weighted) / 7
                                average_metrics = round(average_metrics, 2)
                                # add performance metrics to df_model_meta using pd.concat with index
                                df_model_meta = pd.concat([df_model_meta, pd.DataFrame([['meta', x+1, algos[x+1], accuracy, precision, recall, roc_auc, gmean, mcc, f1_weighted, log_loss, average_metrics, f'{final_estimator.get_params()}', 'meta']], columns=df_model_meta.columns, index=[x])], axis=0)

                        # Adding average probability for each model

                        df_prob_meta_t = df_prob_meta.transpose()

                        # total number of predictions
                        n_total = df_prob_meta_t.shape[0]

                        # create new pandas series with total number of models
                        prob_meta = []

                        for i in range(11):
                                prob = []
                                # calculate the confidence interval for each algorithm
                                for n in range(n_total):
                                        prob.append(df_prob_meta_t.iloc[n, i])
                                prob_average = np.mean(prob).round(2)
                                # add prob_average to prob_meta
                                prob_meta.append(prob_average)

                        # add prob_meta to df_model_meta as new column "average_probability"
                        df_model_meta['average_probability'] = pd.Series(prob_meta)

                        df_model_meta['rank'] = df_model_meta['overall_performance'] * df_model_meta['average_probability'] / 10000
                        # df_model_meta sort by overall performance and average probability in descending order
                        df_model_meta = df_model_meta.sort_values(by=['rank'], ascending=False)
                        # sort values in df_prob_meta by df_temp index
                        df_prob_meta = df_prob_meta.reindex(df_model_meta.index)
                        # sort values in df_pred_meta by df_temp index
                        df_pred_meta = df_pred_meta.reindex(df_model_meta.index)
                        # sort values in algo by df_temp index
                        algo_meta = [algo[i] for i in df_model_meta.index].copy()
                        # sort values in algo_names by df_temp index
                        algo_names_meta = [algo_names[i] for i in df_model_meta.index].copy()


                        # reset indexes for all dataframes
                        df_model_meta = df_model_meta.reset_index(drop=True)
                        df_prob_meta = df_prob_meta.reset_index(drop=True)
                        df_pred_meta = df_pred_meta.reset_index(drop=True)

                        for x in range(11):
                                df_model_meta.model_id[x] = f'meta_{x+1}'

                        df_model_dict_meta[f'{key_model}_meta'] = df_model_meta
                        df_prob_dict_meta[f'{key_prob}_meta'] = df_prob_meta
                        df_pred_dict_meta[f'{key_pred}_meta'] = df_pred_meta
                        algo_dict[f'{key_algo}'] = algo_meta
                        algo_names_dict[f'{key_name_algo}'] = algo_names_meta

                        # # Save all dataframes to csv
                        # df_model_meta.to_csv(path + f'{key_model}_meta.csv')
                        # df_prob_meta.to_csv(path + f'{key_prob}_meta.csv')
                        # df_pred_meta.to_csv(path + f'{key_pred}_meta.csv')

                        percent_complete = percent_complete +  (1 / len(df_model_dict.keys()))
                        my_bar.progress(percent_complete)

                latest_iteration.text('Calculation of meta-models completed')

                # create empty dataframe
                df_top_rows = pd.DataFrame(columns=df_model_dict_meta['df_model_all_meta'].columns)
                df_top_rows['cluster'] = None

                for key in df_model_dict_meta.keys():

                        string = key.replace('df_model_', '')
                        string = string.replace('_meta', '')

                        # return row from f_model_dict_meta[key] with highest rank
                        highest_rank = df_model_dict_meta[key].sort_values(by=['rank'], ascending=False)['rank'].iloc[0]
                        rank_df = df_model_dict_meta[key][df_model_dict_meta[key]['rank'] == highest_rank]

                        rank_df['cluster'] = string

                        df_top_rows = pd.concat([df_top_rows, rank_df], axis=0)

                # sort by rank
                df_top_rows = df_top_rows.sort_values(by=['rank'], ascending=False)
                # reset index

                df_top_rows = df_top_rows.reset_index(drop=True)
                # create new column by combining rank and index
                df_top_rows['cluster'] = (df_top_rows.index + 1).astype(str) + '_' + df_top_rows['cluster']


                fig = go.Figure()

                # https://plotly.com/python/horizontal-bar-charts/

                metrics = ['accuracy', 'precision', 'recall', 'roc_auc_score', 'geometric_mean_score', 'matthews_corrcoef', 'f1_weighted', 'average_probability']
                metrics_legend = ['Accuracy', 'Precision', 'Recall', 'ROC AUC', 'Geometric Mean', 'Matthews CorrCoeff', 'F1 Score', 'Confidence Interval']
                colors = ['#c6dfcd', '#f5f2d3', '#e6d5c3', '#c9a8a0', '#737d89', '#869f9f', '#a3a0b8', '#a7bed3']

                for i in range (8):
                        value = df_top_rows[f'{metrics[i]}']
                        if metrics[i] != 'average_probability':
                                fig.add_trace(go.Bar(y=df_top_rows.cluster, x=value, name=metrics_legend[i], orientation='h', text= np.round(value/100, 4), marker=dict(color=colors[i], line=dict(color=colors[i], width=3))))
                        else:
                                fig.add_trace(go.Bar(y=df_top_rows.cluster, x=value*7, name=metrics_legend[i], orientation='h', text= np.round(value/100, 4), marker=dict(color=colors[i], line=dict(color=colors[i], width=3))))

                # update x values size
                fig.update_layout(yaxis_tickfont_size=12, xaxis_tickfont_size=12, title_text='Model Performance', title_x=0.5)

                fig.update_layout(barmode='stack', paper_bgcolor='rgb(248, 248, 255)', plot_bgcolor='rgb(248, 248, 255)', margin=dict(l=0, r=0, t=0, b=0), showlegend=True, yaxis=dict(categoryorder = 'total ascending'))

                # add title for x axes and y axes
                fig.update_xaxes(title_text='Overall Performance', title_font=dict(size=14))
                fig.update_yaxes(title_text='Cluster', title_font=dict(size=14))

                # update legend names
                fig.update_layout(legend_title_text='Metrics')
                # move legend outside plot
                fig.update_layout(legend=dict(x=-0.1, y=-0.5))
                fig.update_traces(textposition='inside')
                fig.update_traces(insidetextanchor='middle')
                # add % to text
                fig.update_traces(texttemplate='%{text:.2%}')

                # get total length of bars
                total_length = 0
                for i in fig.data:
                        total_length += i.x[0]

                fig.add_annotation(xref='x', yref='y', x=total_length, y=df_top_rows.shape[0], text='Best Performing model', font = dict(size = 14), showarrow=False)

                for i in range (df_top_rows.shape[0]):
                        fig.add_annotation(xref='x', yref='y', x=total_length, y=df_top_rows.cluster[i], text=df_top_rows.algorithm_name[i], font = dict(size = 12), showarrow=False)

                fig.update_layout(width=total_length, height=df_top_rows.shape[0]*100)

                st.title('Clusters using HDBSCAN')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f'**Clusters:** {len(df_top_rows)}')
                st.markdown(f'**Cluster size:** {df_top_rows.shape[0]}')
                st.markdown(f'**Cluster density:** {round(df_top_rows.shape[0] / df_top_rows.shape[0], 2)}')

                
else:
        st.info('Awaiting for CSV file to be uploaded.')


    













































# umap_chart = st.empty()
# if st.checkbox('Show UMAP Chart'):
# # Apply robust scaler to df_source
#         scaler = preprocessing.RobustScaler()
#         df_source_scaled = scaler.fit_transform(df_source)
#         x_train = pd.DataFrame(df_source_scaled, columns=df_source.columns)
#         df_umap = umap_model(df_probabilities, df_model)
#         df_umap_best = umap_best(df_umap)
#         # Define hyperparameters for each  final estimator, based on top performing model from base layer
#         knn_params = string_to_dict(get_hyperparameters('K-Nearest Neighbor'))
#         svm_params = string_to_dict(get_hyperparameters('Support Vector Machine'))
#         gnb_params = string_to_dict(get_hyperparameters('Gaussian Naive Bayes'))
#         mlp_params = string_to_dict(get_hyperparameters('Multilayer Perceptron'))
#         lr_params = string_to_dict(get_hyperparameters('Logistic Regression'))
#         lda_params = string_to_dict(get_hyperparameters('Linear Discriminant Analysis'))
#         qda_params = string_to_dict(get_hyperparameters('Quadratic Discriminant Analysis'))
#         rf_params = string_to_dict(get_hyperparameters('Random Forest'))
#         et_params = string_to_dict(get_hyperparameters('Extra Trees'))
#         ab_params = string_to_dict(get_hyperparameters('Adaptive Boosting'))
#         gb_params = string_to_dict(get_hyperparameters('Gradient Boosting'))

#         # Define base layer estimators (one per algorithm)
#         # Add estimators
#         estimators = [('knn', KNeighborsClassifier(algorithm=get_value(knn_params, 'algorithm'), metric=get_value(knn_params, 'metric'),
#                 n_neighbors= get_value(knn_params, 'n_neighbors'), weights=get_value(knn_params, 'weights'))),
#                 ('svm', SVC(C=get_value(svm_params, 'C'), kernel=get_value(svm_params, 'kernel'),probability=True)),
#                 ('gnb', GaussianNB(var_smoothing=get_value(gnb_params, 'var_smoothing'))),
#                 ('mlp', MLPClassifier(activation=get_value(mlp_params, 'activation'), alpha=get_value(mlp_params, 'alpha'),
#                 max_iter=get_value(mlp_params, 'max_iter'), solver=get_value(mlp_params, 'solver'), tol=get_value(mlp_params, 'tol'))),
#                 ('lr', LogisticRegression(C=get_value(lr_params, 'C'), max_iter=get_value(lr_params, 'max_iter'),
#                 penalty=get_value(lr_params, 'penalty'), solver=get_value(lr_params, 'solver'))),
#                 ('lda', LinearDiscriminantAnalysis(shrinkage=get_value(lda_params, 'shrinkage'), solver=get_value(lda_params, 'solver'))),
#                 ('qda', QuadraticDiscriminantAnalysis(reg_param=get_value(qda_params, 'reg_param'), tol=get_value(qda_params, 'tol'))),
#                 ('rf', RandomForestClassifier(criterion=get_value(rf_params, 'criterion'), n_estimators=get_value(rf_params, 'n_estimators'))),
#                 ('et', ExtraTreesClassifier(criterion=get_value(et_params, 'criterion'), n_estimators=get_value(et_params, 'n_estimators'))),
#                 ('ab', AdaBoostClassifier(algorithm=get_value(ab_params, 'algorithm'), learning_rate=get_value(ab_params, 'learning_rate'),
#                 n_estimators=get_value(ab_params, 'n_estimators'))),
#                 ('gb', GradientBoostingClassifier(criterion=get_value(gb_params, 'criterion'), learning_rate=get_value(gb_params, 
#                 'learning_rate'), n_estimators=get_value(gb_params, 'n_estimators')))
#                 ]
#         # create dataframe for df_model_meta with columns names from df_model
#         df_model_meta = pd.DataFrame(columns=df_model.columns)

#         # create dataframe for top_models probabilities
#         df_prob_meta = pd.DataFrame()

#         for x in range(0, len(estimators)):
#                 final_estimator = estimators[x][1]
#                 clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator, n_jobs=-1)
#                 clf.fit(x_train, y_train)
#                 y_pred = clf.predict(x_train)
#                 y_pred = pd.Series(y_pred)

#         # probabilities dataframe
#                 y_pred_prob = clf.predict_proba(x_train)
#                 y_pred_prob_df = pd.DataFrame(y_pred_prob, columns=['prob_0', 'prob_1'])
#                 y_pred_prob_df['target'] = y_train
#                 y_pred_prob_df['predicted'] = y_pred
#                 # create new column, set value to prob_0 if target is 0 and prob_1 if target is 1
#                 y_pred_prob_df['pred_prob'] = np.where(y_pred_prob_df['target'] == 0, y_pred_prob_df['prob_0'], y_pred_prob_df['prob_1'])
#                 # remove prob_0 and prob_1 columns, target and rpected columns
#                 y_pred_prob_df = y_pred_prob_df.drop(['prob_0', 'prob_1', 'target', 'predicted'], axis=1)
#                 # transpose the data frame and convert values to %
#                 y_pred_prob_df = y_pred_prob_df.T
#                 y_pred_prob_df = y_pred_prob_df.apply(lambda x: x * 100).round(2)
#                 # set index to x
#                 y_pred_prob_df.index = [x]
#                 # add row to df_prob_meta using pd.concat
#                 df_prob_meta = pd.concat([df_prob_meta, y_pred_prob_df], axis=0)

#         # performance metrics dataframe
#                 accuracy = round(accuracy_score(y_train, y_pred)*100, 2)
#                 precision = round(precision_score(y_train, y_pred, average='weighted')*100, 2)
#                 recall = round(recall_score(y_train, y_pred, average='weighted')*100, 2)
#                 roc_auc = round(roc_auc_score(y_train, y_pred, average='weighted')*100, 2)
#                 gmean = round(geometric_mean_score(y_train, y_pred, average='weighted')*100, 2)
#                 mcc = round(matthews_corrcoef(y_train, y_pred)*100, 2)
#                 f1_weighted = round(f1_score(y_train, y_pred, average='weighted')*100, 2)
#                 log_loss = round(metrics.log_loss(y_train, y_pred)*100, 2)
#                 average_metrics = (accuracy + precision + recall + roc_auc + gmean + mcc + f1_weighted) / 7
#                 average_metrics = round(average_metrics, 2)
#                 # add performance metrics to df_model_meta using pd.concat with index
#                 df_model_meta = pd.concat([df_model_meta, pd.DataFrame([[f'meta_{x+1}', x+1, accuracy, precision, recall, roc_auc, gmean, 
#                                 mcc, f1_weighted, log_loss, average_metrics, f'{final_estimator.get_params()}']], 
#                                 columns=df_model_meta.columns, index=[x])], axis=0)

#         # create umap dataframe for metamodels predictions
#         df_umap_meta = umap_model(df_prob_meta, df_model_meta)

#         # concat df_umap and df_umap_meta
#         df_umap_all = pd.concat([df_umap, df_umap_meta], axis=0)

#         with umap_chart.container():
#                 fig = umap_plot(df_umap_all)
# else:
#         umap_chart.empty()
