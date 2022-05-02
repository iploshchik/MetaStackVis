
import pandas as pd
import umap
from sklearn import preprocessing
import plotly.express as px
import numpy as np
import streamlit as st
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

# Page layout
st.set_page_config(page_title='Visually-Assisted Performance Evaluation of Metamodels in Stacking Ensemble Learning',layout='wide')
st.subheader('Visually-Assisted Performance Evaluation of Metamodels in Stacking Ensemble Learning')


# Sidebar - Collects user input features into dataframes
# st.sidebar.subheader('Upload your original dataset')
uploaded_source = st.sidebar.file_uploader("Upload your original dataset", type=["csv"], key='source')
# st.sidebar.subheader('Upload your target data')
uploaded_target = st.sidebar.file_uploader("Upload your target data", type=["csv"], key='target')
# st.sidebar.subheader('Upload your model probabilities data')
uploaded_probabilities = st.sidebar.file_uploader("Upload your model probabilities data", type=["csv"], key='probabilities')
# st.sidebar.subheader('Upload your model data')
uploaded_model = st.sidebar.file_uploader("Upload your model data", type=["csv"], key='model')


# Sidebar - Specify parameter settings
st.sidebar.subheader('Set Parameters for UMAP Chart')
parameter_umap_n_neighbors = st.sidebar.number_input('Number of neighbors (n_neighbors)', 5, key='n_neighbors')
parameter_umap_metric = st.sidebar.selectbox('Metric', ('euclidean', 'manhattan', 'chebyshev', 'minkowski'), key='metric')
parameter_umap_min_dist = st.sidebar.number_input('Minimal distance', 0.1, key='min_dist')
st.sidebar.write('---')
st.sidebar.subheader('General Plotly charts Parameters')
plotly_size = st.sidebar.number_input('Size', 600, key='size')
# plotly_color = st.sidebar.selectbox('Color', ('red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'grey'), key='color')
st.sidebar.write('---')
st.write('Choose visualization options below')


# UMAP function
def umap_model (df_prob=pd.read_csv(uploaded_probabilities), df_mod=pd.read_csv(uploaded_model), parameter_umap_n_neighbors = 10, parameter_umap_min_dist = 0.5, 
                parameter_umap_metric = 'euclidean'):

        algos = {1:'K-Nearest Neighbor', 2:'Support Vector Machine', 3:'Gaussian Naive Bayes', 4:'Multilayer Perceptron', 5:'Logistic Regression',
        6:'Linear Discriminant Analysis', 7:'Quadratic Discriminant Analysis', 8:'Random Forest', 9:'Extra Trees', 10:'Adaptive Boosting',
        11:'Gradient Boosting'}

        umap_model = umap.UMAP(n_neighbors=parameter_umap_n_neighbors, metric=parameter_umap_metric, min_dist=parameter_umap_min_dist)
        umap_embedding = umap_model.fit_transform(df_prob)
        #convert umap_embedding to dataframe
        df_umap = pd.DataFrame(umap_embedding, columns=['UMAP_1', 'UMAP_2'])
        # Add algortim number (keeping in mind the same row structure in topModels.csv and topModelsProbabilities.csv)
        df_umap['algorithm_id'] = df_mod['algorithm_id']
        df_umap['algorithm_name'] = df_umap['algorithm_id'].map(algos)
        df_umap['model_id'] = df_mod['model_id']
        # add hyperparameters column
        df_umap['hyperparameters'] = df_mod['params']
        # Add model specific metrics
        df_umap['accuracy'] = df_model['mean_test_accuracy']
        df_umap['precision'] = df_model['mean_test_precision_weighted']
        df_umap['recall'] = df_model['mean_test_recall_weighted']
        df_umap['roc_auc_score'] = df_model['mean_test_roc_auc_ovo_weighted']
        df_umap['geometric_mean_score'] = df_model['geometric_mean_score_weighted']
        df_umap['matthews_corrcoef'] = df_model['matthews_corrcoef']
        df_umap['f1_weighted'] = df_model['f1_weighted']
        df_umap['log_loss'] = df_model['log_loss']
        df_umap['performance'] = round((df_umap['accuracy'] + df_umap['precision'] + df_umap['recall'] + df_umap['roc_auc_score'] + \
                        df_umap['geometric_mean_score'] + df_umap['matthews_corrcoef'] + df_umap['f1_weighted']) / 7, 2)
        # create new column for text of points
        df_umap['text'] = df_umap['algorithm_name'] + '<br>' + 'Performance: ' + \
                df_umap['performance'].astype(str) + '%' + '<br>' + 'Model ID: ' + df_umap['model_id'].astype(str) + \
                '<br>' + 'Accuracy: ' + df_umap['accuracy'].astype(str) + '%' + '<br>' + 'Precision: ' + \
                df_umap['precision'].astype(str) + '%' + '<br>' + 'Recall: ' + df_umap['recall'].astype(str) + \
                '%' + '<br>' + 'ROC AUC: ' + df_umap['roc_auc_score'].astype(str) + '<br>' + 'Geometric Mean: ' + \
                df_umap['geometric_mean_score'].astype(str) + '<br>' + 'Matthews Correlation: ' + \
                df_umap['matthews_corrcoef'].astype(str) + '<br>' + 'F1: ' + df_umap['f1_weighted'].astype(str) + \
                '<br>' + 'Log Loss: ' + df_umap['log_loss'].astype(str)
        # drop metrics that are not needed
        df_umap = df_umap.drop(columns=['accuracy', 'precision', 'recall', 'roc_auc_score', 'geometric_mean_score', 
                                        'matthews_corrcoef', 'f1_weighted', 'log_loss'])
        # convert columns types
        df_umap = df_umap.astype({'UMAP_1': 'float64', 'UMAP_2': 'float64', 'performance': 'float64', 'algorithm_id': 'int64',
                                    'algorithm_name': 'str', 'model_id': 'str', 'hyperparameters': 'str', 'text': 'str'})

        return df_umap

# Function to choose umap algorithmes based on perfromance (top 11 models)
def umap_best(df_umap):
   # Select hyperparameters for best model in each algorithm
    df_umap_best = df_umap.groupby('algorithm_id').apply(lambda x: x.sort_values('performance', ascending=False).iloc[0])
    # reset algorithm_nr as  index
    df_umap_best = df_umap_best.reset_index(drop=True)
    # keep only algorithm number, name, performance and hyperparameters
    df_umap_best = df_umap_best[['algorithm_id', 'algorithm_name', 'performance', 'hyperparameters']]
    return df_umap_best 

# Supporting functions to extract hyperparameters values

# Return hyperparameters based on algorithm name from df_umap_best
def get_hyperparameters(algorithm_name):
    return df_umap_best[df_umap_best['algorithm_name'] == algorithm_name]['hyperparameters'].values[0]

# return value of key in dictionary
def get_value(dictionary, key):
    return dictionary[key]

# convert string to dictionary
def string_to_dict(string):
    return ast.literal_eval(string)

# UMAP plot function
def umap_plot(df_umap):
        # Define symbols for each algorithm
        symbols = ['circle', 'square', 'x', 'cross', 'diamond', 'star', 'hexagram', 'triangle-right', 'triangle-left', 'triangle-down', 'triangle-up']
        # Plot UMAP, add hovertext and symbols, define colorscale by performance, add title
        fig = px.scatter(df_umap, x='UMAP_1', y='UMAP_2', color='performance', hover_name='text',
                symbol = df_umap['algorithm_id'], symbol_sequence = symbols, labels=dict(UMAP_1='', UMAP_2='', performance='Performance'),
                color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_layout(title_text='UMAP Plot')
        fig.update_layout(showlegend=False)
        # Set marker symbol shape based on algorithm
        fig.update_traces(marker=dict(size=10, opacity=0.9, line=dict(width=1, color='Black')), selector=dict(mode='markers'))
        # Remove axis labels
        fig.update_layout(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
        # define plot as square
        fig.update_layout(width=600, height=600)
        # add tooltip
        fig.update_layout(hovermode='closest')
        st.plotly_chart(fig)


#------------------------------------------------------------------------------------------------


if uploaded_probabilities is not None and uploaded_model is not None and uploaded_target is not None \
            and uploaded_target is not None:
    df_probabilities = pd.read_csv(uploaded_probabilities)
    df_model = pd.read_csv(uploaded_model)
    df_target = pd.read_csv(uploaded_target)
    df_source = pd.read_csv(uploaded_source)
    y_train = df_target['class']
    target = df_target['class'].tolist()
    algo_nr = df_model['algorithm_id']
    # Check for missing values and print values per column if any are missing for each of dataframes
    if df_probabilities.isnull().values.any() or df_model.isnull().values.any() or df_target.isnull().values.any() or df_source.isnull().values.any():
        st.write('Missing values in dataframes')
        st.write(df_probabilities.isnull().sum())
        st.write(df_model.isnull().sum())
        st.write(df_target.isnull().sum())
        st.write(df_source.isnull().sum())
    # Check if there are any duplicate rows in dataframes
    if df_probabilities.duplicated().any() or df_model.duplicated().any() or df_target.duplicated().any() or df_source.duplicated().any():
        st.write('Duplicate rows in dataframes')
        st.write(df_probabilities.duplicated().sum())
        st.write(df_model.duplicated().sum())
        st.write(df_target.duplicated().sum())
        st.write(df_source.duplicated().sum())
    if st.checkbox('Show UMAP Chart'):
        # Apply robust scaler to df_source
        scaler = preprocessing.RobustScaler()
        df_source_scaled = scaler.fit_transform(df_source)
        x_train = pd.DataFrame(df_source_scaled, columns=df_source.columns)
        df_umap = umap_model(df_probabilities, df_model)
        df_umap_best = umap_best(df_umap)
        # Define hyperparameters for each  final estimator, based on top performing model from base layer
        knn_params = string_to_dict(get_hyperparameters('K-Nearest Neighbor'))
        svm_params = string_to_dict(get_hyperparameters('Support Vector Machine'))
        gnb_params = string_to_dict(get_hyperparameters('Gaussian Naive Bayes'))
        mlp_params = string_to_dict(get_hyperparameters('Multilayer Perceptron'))
        lr_params = string_to_dict(get_hyperparameters('Logistic Regression'))
        lda_params = string_to_dict(get_hyperparameters('Linear Discriminant Analysis'))
        qda_params = string_to_dict(get_hyperparameters('Quadratic Discriminant Analysis'))
        rf_params = string_to_dict(get_hyperparameters('Random Forest'))
        et_params = string_to_dict(get_hyperparameters('Extra Trees'))
        ab_params = string_to_dict(get_hyperparameters('Adaptive Boosting'))
        gb_params = string_to_dict(get_hyperparameters('Gradient Boosting'))

        # Define base layer estimators (one per algorithm)
        # Add estimators
        estimators = [('knn', KNeighborsClassifier(algorithm=get_value(knn_params, 'algorithm'), metric=get_value(knn_params, 'metric'),
                n_neighbors= get_value(knn_params, 'n_neighbors'), weights=get_value(knn_params, 'weights'))),
                ('svm', SVC(C=get_value(svm_params, 'C'), kernel=get_value(svm_params, 'kernel'),probability=True)),
                ('gnb', GaussianNB(var_smoothing=get_value(gnb_params, 'var_smoothing'))),
                ('mlp', MLPClassifier(activation=get_value(mlp_params, 'activation'), alpha=get_value(mlp_params, 'alpha'),
                max_iter=get_value(mlp_params, 'max_iter'), solver=get_value(mlp_params, 'solver'), tol=get_value(mlp_params, 'tol'))),
                ('lr', LogisticRegression(C=get_value(lr_params, 'C'), max_iter=get_value(lr_params, 'max_iter'),
                penalty=get_value(lr_params, 'penalty'), solver=get_value(lr_params, 'solver'))),
                ('lda', LinearDiscriminantAnalysis(shrinkage=get_value(lda_params, 'shrinkage'), solver=get_value(lda_params, 'solver'))),
                ('qda', QuadraticDiscriminantAnalysis(reg_param=get_value(qda_params, 'reg_param'), tol=get_value(qda_params, 'tol'))),
                ('rf', RandomForestClassifier(criterion=get_value(rf_params, 'criterion'), n_estimators=get_value(rf_params, 'n_estimators'))),
                ('et', ExtraTreesClassifier(criterion=get_value(et_params, 'criterion'), n_estimators=get_value(et_params, 'n_estimators'))),
                ('ab', AdaBoostClassifier(algorithm=get_value(ab_params, 'algorithm'), learning_rate=get_value(ab_params, 'learning_rate'),
                n_estimators=get_value(ab_params, 'n_estimators'))),
                ('gb', GradientBoostingClassifier(criterion=get_value(gb_params, 'criterion'), learning_rate=get_value(gb_params, 
                'learning_rate'), n_estimators=get_value(gb_params, 'n_estimators')))
                ]
        # create dataframe for df_model_meta with columns names from df_model
        df_model_meta = pd.DataFrame(columns=df_model.columns)

        # create dataframe for top_models probabilities
        df_prob_meta = pd.DataFrame()

        for x in range(0, len(estimators)):
                final_estimator = estimators[x][1]
                clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator, n_jobs=-1)
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
                # remove prob_0 and prob_1 columns, target and rpected columns
                y_pred_prob_df = y_pred_prob_df.drop(['prob_0', 'prob_1', 'target', 'predicted'], axis=1)
                # transpose the data frame and convert values to %
                y_pred_prob_df = y_pred_prob_df.T
                y_pred_prob_df = y_pred_prob_df.apply(lambda x: x * 100).round(2)
                # set index to x
                y_pred_prob_df.index = [x]
                # add row to df_prob_meta using pd.concat
                df_prob_meta = pd.concat([df_prob_meta, y_pred_prob_df], axis=0)

        # performance metrics dataframe
                accuracy = round(accuracy_score(y_train, y_pred)*100, 2)
                precision = round(precision_score(y_train, y_pred, average='weighted')*100, 2)
                recall = round(recall_score(y_train, y_pred, average='weighted')*100, 2)
                roc_auc = round(roc_auc_score(y_train, y_pred, average='weighted')*100, 2)
                gmean = round(geometric_mean_score(y_train, y_pred, average='weighted')*100, 2)
                mcc = round(matthews_corrcoef(y_train, y_pred)*100, 2)
                f1_weighted = round(f1_score(y_train, y_pred, average='weighted')*100, 2)
                log_loss = round(metrics.log_loss(y_train, y_pred)*100, 2)
                average_metrics = (accuracy + precision + recall + roc_auc + gmean + mcc + f1_weighted) / 7
                average_metrics = round(average_metrics, 2)
                # add performance metrics to df_model_meta using pd.concat with index
                df_model_meta = pd.concat([df_model_meta, pd.DataFrame([[f'meta_{x+1}', x+1, accuracy, precision, recall, roc_auc, gmean, 
                                mcc, f1_weighted, log_loss, average_metrics, f'{final_estimator.get_params()}']], 
                                columns=df_model_meta.columns, index=[x])], axis=0)

        # create umap dataframe for metamodels predictions
        df_umap_meta = umap_model(df_prob_meta, df_model_meta)

        # concat df_umap and df_umap_meta
        df_umap_all = pd.concat([df_umap, df_umap_meta], axis=0)

        fig = umap_plot(df_umap_all)

        with st.expander("Algorithm Details"):
            st.write('Following algorithms are presented in the chart: K-Nearest Neighbor, Support Vector Machine, \
            Gaussian Naive Bayes, Multilayer Perceptron,Logistic Regression, Linear Discriminant Analysis, \
            Quadratic Discriminant Analysis, Random Forest, Extra Trees, Adaptive Boosting, Gradient Boosting')
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        path = r'D:\github\2dv50e\Data\1. Heart Disease'
        df_probabilities = pd.read_csv(path + r'\topModelsProbabilities.csv')
        df_model = pd.read_csv(path + r'\topModels.csv')
        algo_nr = df_model['algorithm_id']
        if st.checkbox('Show UMAP Chart'):