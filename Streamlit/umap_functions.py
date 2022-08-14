# import libraries
import pandas as pd
import umap
import plotly.graph_objects as go
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
import math
from sklearn.preprocessing import minmax_scale
import streamlit as st

symbols = ['circle', 'square', 'x', 'cross', 'diamond', 'star', 'hexagram', 'triangle-right', 'triangle-left', 'triangle-down', 'triangle-up']
algos = {1:'K-Nearest Neighbor', 2:'Support Vector Machine', 3:'Gaussian Naive Bayes', 4:'Multilayer Perceptron', 5:'Logistic Regression', 6:'Linear Discriminant Analysis', 7:'Quadratic Discriminant Analysis', 8:'Random Forest', 9:'Extra Trees', 10:'Adaptive Boosting', 11:'Gradient Boosting'}
algo = ['knn', 'svm', 'gnb', 'mlp', 'lr', 'lda', 'qda', 'rf', 'et', 'ab', 'gb']

# Button layout for UMAP
def create_layout_button(n, neighb = 8):
        list = [False] * (neighb*11 + 1)
        list[-1] = True
        if n == 0:
                list[n] = True
                for i in range(1, 11):
                        list[i] = True
        else:
                list[11*n] = True
                for i in range(11*n, 11*n + 11):
                        list[i] = True
        return dict(label = f'number of neighbours {n+3}',
                        method = 'restyle',
                        args = [{'visible': list}])



@st.experimental_memo
def plottingUMAP(df_model, df_model_meta, df_prob, df_prob_meta):
        # concatinate df_model and df_model_meta
        df_model_all = pd.concat([df_model, df_model_meta], axis=0)
        df_model_all = df_model_all.astype({'model_id': 'str', 'algorithm_id': 'int64', 'accuracy': 'float64', 'precision': 'float64', 
                                'recall': 'float64', 'roc_auc_score': 'float64', 'geometric_mean_score': 'float64', 
                                'matthews_corrcoef': 'float64', 'f1_weighted': 'float64', 'log_loss': 'float64', 
                                'overall_performance': 'float64', 'average_probability': 'float64'})
        # scale avarage probability in scale 0 to 1
        df_model_all['average_probability_norm'] = np.round(minmax_scale(df_model_all['average_probability'], feature_range=(0.2, 1)), 1)

        # create new column "size", set to 2 for rows with "meta" in "model_id", else 1
        df_model_all['size'] = np.where(df_model_all['model_id'].str.contains('meta'), 2, 1)
        # create new column for text of points
        df_model_all['text'] = df_model_all['algorithm_name'] + '<br>' + 'Performance: ' + \
                df_model_all['overall_performance'].astype(str) + '%' + '<br>' + 'Model ID: ' + df_model_all['model_id'].astype(str) + \
                '<br>' + 'Accuracy: ' + df_model_all['accuracy'].astype(str) + '%' + '<br>' + 'Precision: ' + \
                df_model_all['precision'].astype(str) + '%' + '<br>' + 'Recall: ' + df_model_all['recall'].astype(str) + \
                '%' + '<br>' + 'ROC AUC: ' + df_model_all['roc_auc_score'].astype(str) + '<br>' + 'Geometric Mean: ' + \
                df_model_all['geometric_mean_score'].astype(str) + '<br>' + 'Matthews Correlation: ' + \
                df_model_all['matthews_corrcoef'].astype(str) + '<br>' + 'F1: ' + df_model_all['f1_weighted'].astype(str) + \
                '<br>' + 'Average Probability: ' + df_model_all['average_probability'].astype(str)

        # set df_prob_meta columns as df_prob columns
        df_prob_meta.columns = df_prob.columns

        # concatinate df_prob and df_prob_meta
        df_prob_all = pd.concat([df_prob, df_prob_meta], axis=0)

        ######################################################################################################################

        ### UMAP dimension reduction algorithm


        # convert symbols to dictionary with keys from 1 to 11
        symbols_dict = dict(zip(range(1, 12), symbols))

        # define empty dictionary for UMAP layout
        umap_figs = {}


        for i in [3, 4, 5, 6]:
                for j in [0.2, 0.5]:
                        for metric in ['euclidean', 'manhattan']:
                                df_model_all_umap = df_model_all.copy()
                                df_prob_all_umap = df_prob_all.copy()
                                umap_model = umap.UMAP(n_neighbors=i, min_dist=j, metric=metric,)
                                #fit transform and convert to dataframe
                                umap_prob = pd.DataFrame(umap_model.fit_transform(df_prob_all_umap), columns=['UMAP_1', 'UMAP_2'])
                                umap_prob.index = df_prob_all_umap.index
                                df_model_all_umap = pd.concat([df_model_all_umap, umap_prob], axis=1)

                                fig = go.Figure()

                                for key in algos.keys():
                                        df_model_red = df_model_all_umap[df_model_all_umap['algorithm_id'] == key]
                                        fig.add_trace(go.Scatter(x=df_model_red['UMAP_1'], y=df_model_red['UMAP_2'], mode='markers', hovertext=df_model_red['text'], marker=dict(size=df_model_red['size']*20, symbol = df_model_red['algorithm_id'].map(symbols_dict), opacity = df_model_red['average_probability_norm'], line=dict(width=1, color='Black'), color=df_model_red['overall_performance'], coloraxis='coloraxis'), name = algos[key]))

                                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0))
                                
                                # remove axes labels
                                fig.update_xaxes(showticklabels=False)
                                fig.update_yaxes(showticklabels=False)

                                fig.update_traces(selector=dict(mode='markers'))
                                # define symbols for markers based on symbols defined
                                fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Rockwell"), hovermode='closest')
                                # add tooltip
                                fig.update_layout(coloraxis=dict(showscale=True, colorscale='Viridis'), showlegend=True)
                                    # add title to colorbar
                                fig.update_layout(coloraxis_colorbar=dict(title='Metric-Based Performance', titleside='right'))
                                # increase legend size
                                fig.update_layout(legend_title = 'Algorithm', legend=dict(x=-0.4, y=1, font=dict(size=12)))

                                min_perf = math.floor(df_model_all_umap['overall_performance'].min()/5)*5
                                max_perf = math.ceil(df_model_all_umap['overall_performance'].max()/5)*5

                                # set min and max value for legend as min and max value of overall_performance
                                fig.update_layout(coloraxis=dict(cmin=min_perf, cmax=max_perf))

                                # define subplot size
                                fig.update_layout(width=600, height=600, margin=dict(l=0, r=0, t=0, b=0))

                                # add fig to dictionary
                                umap_figs[f'UMAP_{i}_{j}_{metric}'] = fig

        return umap_figs
