# import libraries
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
import math
from umap_functions import *
from coverage_function import *
from supporting_functions import *
from plotting_comparison import *


# define random seed
np.random.seed(42)

# Page layout
st.header('Visually-Assisted Performance Evaluation of Metamodels in Stacking Ensemble Learning')
st.header('Plotting')


st.sidebar.subheader('Set Parameters for UMAP Chart')
parameter_umap_n_neighbors = st.sidebar.selectbox('Number of neighbors (n_neighbors)', [3, 4, 5, 6], key='n_neighbors')
parameter_umap_metric = st.sidebar.selectbox('Metric', ['euclidean', 'manhattan'], key='metric')
parameter_umap_min_dist = st.sidebar.selectbox('Minimal distance', [0.2, 0.5],key='min_dist')

# add varibales from the previous page
df_model = st.session_state.df_model
df_prob = st.session_state.df_prob
final_estimators = st.session_state.final_estimators
x_train = st.session_state.x_train
y_train = st.session_state.y_train
x_test = st.session_state.x_test
y_test = st.session_state.y_test
algos = st.session_state.algos
algo = st.session_state.algo
algo_names = st.session_state.algo_names

if st.button('Calculate metamodel performance for initial run or in case source data or clusters have been updated'):

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
                        clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator, n_jobs=-1, cv=None)
                        clf.fit(x_train, y_train)
                        y_pred = clf.predict(x_test)
                        y_pred = pd.Series(y_pred)

                        # probabilities dataframe
                        y_pred_prob = clf.predict_proba(x_test)
                        y_pred_prob_df = pd.DataFrame(y_pred_prob, columns=['prob_0', 'prob_1'])
                        y_pred_prob_df['target'] = y_test
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
                        accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
                        precision = round(precision_score(y_test, y_pred, average='weighted')*100, 2)
                        recall = round(recall_score(y_test, y_pred, average='weighted')*100, 2)
                        roc_auc = round(roc_auc_score(y_test, y_pred, average='weighted')*100, 2)
                        gmean = round(geometric_mean_score(y_test, y_pred, average='weighted')*100, 2)
                        mcc = round(matthews_corrcoef(y_test, y_pred)*100, 2)
                        f1_weighted = round(f1_score(y_test, y_pred, average='weighted')*100, 2)
                        log_loss = round(metrics.log_loss(y_test, y_pred, normalize=True)*100, 2)
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

        st.session_state['df_model_dict'] = df_model_dict
        st.session_state['df_prob_dict'] = df_prob_dict
        st.session_state['df_pred_dict'] = df_pred_dict
        st.session_state['df_model_dict_meta'] = df_model_dict_meta
        st.session_state['df_prob_dict_meta'] = df_prob_dict_meta
        st.session_state['df_pred_dict_meta'] = df_pred_dict_meta
        st.session_state['algo_dict'] = algo_dict
        st.session_state['algo_names_dict'] = algo_names_dict
        st.session_state['df_top_rows'] = df_top_rows


df_model_dict = st.session_state['df_model_dict']
df_prob_dict = st.session_state['df_prob_dict']
df_pred_dict = st.session_state['df_pred_dict']
df_model_dict_meta = st.session_state['df_model_dict_meta']
df_model_dict_meta= st.session_state.df_model_dict_meta
df_prob_dict_meta= st.session_state.df_prob_dict_meta
df_pred_dict_meta = st.session_state.df_pred_dict_meta
algo_dict = st.session_state.algo_dict
algo_names_dict = st.session_state.algo_names_dict
df_top_rows = st.session_state.df_top_rows

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

# make y axes ticks clickable
fig.update_layout(
        yaxis=dict(
                tickmode='array',
                tickvals=df_top_rows.cluster,
                ticktext=df_top_rows.cluster,
                tickangle=0,
                tickfont=dict(
                        family='Arial',
                        size=12,
                        color='black'
                ),
                showgrid=False,
                showline=False,
                showticklabels=True,
                autorange=True,
                domain=[0, 1]
        ))

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

min_performance_all = []
max_performance_all = []

for key_model in df_model_dict:
    # retrun min axn max overall perfromance
    min = df_model_dict[key_model]['overall_performance'].min()
    max = df_model_dict[key_model]['overall_performance'].max()
    # add to list
    min_performance_all.append(min)
    max_performance_all.append(max)
# convert to numpy array
min_performance_all = np.array(min_performance_all)
max_performance_all = np.array(max_performance_all)
# return min value from min performance list and max value from max performance list
min_performance = min_performance_all.min()
# round down min value to closest ten
min_performance= math.floor(min_performance/10)*10
max_performance = max_performance_all.max()
# round up max value to closest ten
max_performance= math.ceil(max_performance/10)*10

for key_model, key_model_meta, key_prob, key_prob_meta, cluster in zip(df_model_dict, df_model_dict_meta, df_prob_dict, df_prob_dict_meta, df_top_rows.cluster):
        with st.spinner('UMAP chart generation'):
                umap_layout = plottingUMAP(df_model_dict[key_model], df_model_dict_meta[key_model_meta], df_prob_dict[key_prob], df_prob_dict_meta[key_prob_meta])
                for key in umap_layout.keys():
                        st.session_state[f'fig_{key_model}_{key}'] = umap_layout[key]

for key_model_meta, key_prob_meta, key_dict in zip(df_model_dict_meta, df_prob_dict_meta, algo_dict):
        with st.spinner('Scatter plot generation'):
                fig = plotting_comparison(df_model_dict_meta[key_model_meta], df_prob_dict_meta[key_prob_meta], algo_dict[key_dict])
                st.session_state[f'fig_{key_model_meta}_comparison'] = fig

      
st.sidebar.subheader('Cluster set-up')
option = st.sidebar.selectbox('Select cluster', df_model_dict.keys(), key='cluster_button')

col1, col2 = st.columns(2)

with col1:
        fig = st.session_state[f'fig_{option}_UMAP_{parameter_umap_n_neighbors}_{parameter_umap_min_dist}_{parameter_umap_metric}']
        st.subheader(f'UMAP chart for {option}')
        st.plotly_chart(fig, use_container_width=True)


with col2:
        fig = st.session_state[f'fig_{option}_meta_comparison']
        st.subheader(f'Comparison chart for {option}')
        st.plotly_chart(fig, use_container_width=True)