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

@st.experimental_memo
def umap_model(df, parameter_umap_n_neighbors = 5, parameter_umap_min_dist =  0.5, parameter_umap_metric = 'euclidean'):
        umap_model = umap.UMAP(n_neighbors=parameter_umap_n_neighbors, metric=parameter_umap_metric, min_dist=parameter_umap_min_dist)
        #fit transform and convert to dataframe
        df_umap = pd.DataFrame(umap_model.fit_transform(df), columns=['UMAP_1', 'UMAP_2'])
        # add index to df_umap
        df_umap.index = df.index
        return df_umap


# Button layout for UMAP
@st.experimental_memo
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
def plottingUMAP(df_model, df_model_meta, df_prob, df_prob_meta, symbols=None, algos=None):
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

    # Add umap_prob to df_model_all and rename columns to 'UMAP_1_prob' and 'UMAP_2_prob' for different hyperparameter of number of neighbors

    # define number of neighbors
    neighb = 4

    for i in range(3, 3 + neighb):
        umap_prob = umap_model(df_prob_all, parameter_umap_n_neighbors = i)
        df_model_all = pd.concat([df_model_all, umap_prob], axis=1)
        df_model_all.rename(columns={'UMAP_1': f'UMAP_1_prob_{i}', 'UMAP_2': f'UMAP_2_prob_{i}'}, inplace=True)

    fig = go.Figure()

    # define subplot size
    fig.update_layout(width=1000, height=800)

    # convert symbols to dictionary with keys from 1 to 11
    symbols_dict = dict(zip(range(1, 12), symbols))

    # Plot UMAP, add hovertext and symbols, define colorscale by performance, add title
    for i in range(3, 3 + neighb):
        for key in algos.keys():
            df_model_red = df_model_all[df_model_all['algorithm_id'] == key]
            fig.add_trace(go.Scatter(x=df_model_red[f'UMAP_1_prob_{i}'], y=df_model_red[f'UMAP_2_prob_{i}'], mode='markers', hovertext=df_model_red['text'], 
                        marker=dict(size=df_model_red['size']*20, symbol = df_model_red['algorithm_id'].map(symbols_dict),
                        opacity = df_model_red['average_probability_norm'], line=dict(width=1, color='Black'),
                        color=df_model_red['overall_performance'], coloraxis='coloraxis'), name = algos[key]))

    # show symbols for each algorithm in the legend

    fig.update_layout(updatemenus=[go.layout.Updatemenu(active = 0, buttons = [create_layout_button(i) for i in range(neighb)],
                    direction = 'down', showactive = False)], legend_title = 'Algorithm', legend=dict(x=-0.4, y=0.9, traceorder='normal'))

    # remove axes labels
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig.update_traces(selector=dict(mode='markers'))
    # define symbols for markers based on symbols defined
    fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=14, font_family="Rockwell"), hovermode='closest')
    # add tooltip
    fig.update_layout(coloraxis=dict(colorscale='Viridis'), showlegend=True)

    min_perf = math.floor(df_model_all['overall_performance'].min()/10)*10
    max_perf = math.ceil(df_model_all['overall_performance'].max()/10)*10

    # set min and max value for legend as min and max value of overall_performance
    fig.update_layout(coloraxis=dict(cmin=min_perf, cmax=max_perf))
    return fig