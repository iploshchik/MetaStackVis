# import libraries
import streamlit as st
import pandas as pd
from sklearn import preprocessing
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
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from umap_functions import *
from coverage_function import *
from supporting_functions import *
from plotting_comparison import *


# define random seed
np.random.seed(42)

# Page layout

st.set_page_config(
    page_title='Visually-Assisted Metamodels Evaluation',
    page_icon="data-science.png")


st.header('Visually-Assisted Performance Evaluation of Metamodels in Stacking Ensemble Learning')


with st.expander('Initital information'):
        st.markdown(
        '''
                Stacking methods (or stacked generalizations) refer to a group of ensemble learning methods where several base models (“weak learners”) are trained and combined into a metamodel with improved predictive power. Stacked generalization can reduce the bias and decrease the generalization error when compared to the use of single learning algorithms. This work is based on data from the visual analytics system, called StackGenVis, developed by ISOVIS research group at LNU and requires input data from that tool, please find more infromation about the tool [here](https://github.com/angeloschatzimparmpas/StackGenVis).  

                This tool can be considered as futher developemnt of the original StackGenVis tool as it provides ens user the possibility to further investigate the impact of different clusters of based models to the overall performance of the metamodel in stacked generalization using HDBSCAN cluster comparison, UMAP and coverage analysis.  

        '''
        )

with st.expander('Usage information'):
                st.markdown(
        '''
                #### The tool is based on the following steps:
                1. Load data from StackGenVis
                2. Preprocess data
                3. Train and plot HDBSCAN clusters based on overall cluster performance (ranking)
                4. Pick the cluster to continue the stacked generalization
                5. Plot UMAP basemodels and metamodels for the selected cluster
                6. Plot coverage analysis of all metamodels for specific cluster  

                Please proceedd with uploading the data from StackGenVis to the respective fields on the side menu and then proceed to the next step by clicking on the "HDBScan Clustering" page on the side menu.
        '''
        )

with st.expander('Credits and author of the tool'):
                st.markdown(
        '''       
                This work is done as a part of 15 sp Bachelor's thesis  at CS LNU in 2022.

                #### Author
                Ilya Ploshchik
                #### Supervisors
                Angelos Chatzimparmpas and Prof. Dr. Andreas Kerren

                ---
                ***Linnaeus university, Faculty of Technology - Spring 2022***
        '''
        )





def on_click_hdb():
        if st.session_state.use_settings:
                st.session_state.hdb_min_cluster_size = min_cluster_size
                st.session_state.hdb_min_samples = min_samples
                st.session_state.hdb_metric = metric


# Sidebar - Collects user input features into dataframes
uploaded_source_train = st.sidebar.file_uploader("Upload your source train data", type=["csv"], key='source_train')
uploaded_source_test = st.sidebar.file_uploader("Upload your source test data", type=["csv"], key='source_test')
uploaded_target_train = st.sidebar.file_uploader("Upload your target train data", type=["csv"], key='target_train')
uploaded_target_test = st.sidebar.file_uploader("Upload your target test data", type=["csv"], key='target_test')
uploaded_probabilities = st.sidebar.file_uploader("Upload your model probabilities data", type=["csv"], key='probabilities')
uploaded_model = st.sidebar.file_uploader("Upload your model data", type=["csv"], key='model')


#------------------------------------------------------------------------------------------------

if uploaded_source_train is not None and uploaded_source_test is not None and uploaded_target_train is not None and         uploaded_target_test is not None and uploaded_probabilities is not None and uploaded_model is not None:

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
                

        # add varibales to session state for use in the other script
        st.session_state['df_model'] = df_model
        st.session_state['df_prob'] = df_prob
        st.session_state['final_estimators'] = final_estimators
        st.session_state['x_train'] = x_train
        st.session_state['y_train'] = y_train
        st.session_state['x_test'] = x_test
        st.session_state['y_test'] = y_test
        st.session_state['algos'] = algos
        st.session_state['algo'] = algo
        st.session_state['algo_names'] = algo_names


# Attribues:
# <a href="https://iconscout.com/icons/data-science" target="_blank">Data Science Icon</a> by <a href="https://iconscout.com/contributors/kiran-shastry" target="_blank">Kiran Shastry</a> from <a href="https://iconscout.com" target="_blank">Iconscout</a>


