# import libraries
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import os
from PIL import Image
import math
from sklearn.preprocessing import minmax_scale
import shutil
import streamlit as st
from coverage_function import *

'''
Probability average (deviation difference) for model probabilities for all models – confidence levels 
Following approach presented: 
- get probabilities for all 11 metamodels
- then comparing 2 metamodels, find the one with better probability of correct class for each instance
- add this value as top probability for each instance and calculate the mean value for all instances for both models

Function operates only with prediction results, which are not the same for the all 11 meta models; the main idea 
is to investigate the differencies and combinations of different metamodels. That can be cosidered as feature engineering 
to df_pred_meta dataframe (dataframe with all 11 meta models predictions per instance) to keep only the columns, 
which are not the same for all 11 meta models.

'''

algos = {1:'K-Nearest Neighbor', 2:'Support Vector Machine', 3:'Gaussian Naive Bayes', 4:'Multilayer Perceptron', 5:'Logistic Regression', 6:'Linear Discriminant Analysis', 7:'Quadratic Discriminant Analysis', 8:'Random Forest', 9:'Extra Trees', 10:'Adaptive Boosting', 11:'Gradient Boosting'}
algo = ['knn', 'svm', 'gnb', 'mlp', 'lr', 'lda', 'qda', 'rf', 'et', 'ab', 'gb']

@st.experimental_memo
def plotting_comparison(df_model_meta, df_prob_meta, algo):

    algo_cap = [i.upper() for i in algo].copy()


    df_prob_meta_t = df_prob_meta.transpose()
    # rename columns to correspond to algorithm names based on perfromance metrics
    df_prob_meta_t.columns = [i for i in df_model_meta.model_id.unique()]

    # total number of predictions
    n_total = df_prob_meta_t.shape[0]

    # create empty list
    prob = []

    df_prob_meta_cor = pd.DataFrame(index=df_prob_meta_t.columns, columns=df_prob_meta_t.columns)

    for i in range(11):
        for j in range(11):
            if i != j:
                prob = []
                # calculate the probability of two models predicting the correct result
                for n in range(n_total):
                    if df_prob_meta_t.iloc[n, i] > df_prob_meta_t.iloc[n, j]:
                        # append df_prob_meta_t.iloc[n, i] to prob
                        prob.append(df_prob_meta_t.iloc[n, i])
                    else:
                        prob.append(df_prob_meta_t.iloc[n, j])
                prob_average = np.mean(prob).round(2)
                # add 2 models average probability to df_prob_meta_cor
                df_prob_meta_cor.loc[f'meta_{i+1}', f'meta_{j+1}'] = prob_average

    df_prob_meta_red = df_prob_meta_t[df_prob_meta_t.apply(lambda x: x.min() < 50, axis=1)]
    # reset index for df_pred_meta_red and df_prob_meta_red
    df_prob_meta_red.reset_index(drop=True, inplace=True)
    # return average probability for rows and store as a new column
    df_prob_meta_red['average_probability'] = df_prob_meta_red.mean(axis=1).round(2) 
    # sort by average_probability_norm
    df_prob_meta_red.sort_values(by='average_probability', ascending=False, inplace=True)
    # drop avergae_probability_norm column
    df_prob_meta_red.drop(columns=['average_probability'], inplace=True)

    # create new dataframe with rows and columns like columns in df_pred_meta_t
    df_pred_meta_cor = pd.DataFrame(index=df_prob_meta_t.columns, columns=df_prob_meta_t.columns)
  
    for i in range(11):
        for j in range(11):
            if i != j:
                # calculate how good two models can contribute to the prediction result
                result = df_prob_meta_t.apply(lambda x: (x[f'meta_{i+1}'] >= 50) or (x[f'meta_{j+1}'] >= 50), axis=1)
                # caclucate the number of wrong predictions for both models
                n_wrong = result[result == False].shape[0]
                # calculate the percentage of wrong predictions in total number of predictions
                perc_correct = round((n_total - n_wrong) / n_total, 4) *100
                # save the percentage in the dataframe
                df_pred_meta_cor.loc[f'meta_{i+1}', f'meta_{j+1}'] = perc_correct
    
    # As Plotly does not support the subplot plotting inside another subplot, we have to save each subplot as a 
    # picture and then load it in the overall plot.

    # create subfolder pictures under path
    path_pictures_prob = './pictures/prob/'
    if not os.path.exists(path_pictures_prob):
        os.makedirs(path_pictures_prob)
    else:
        shutil.rmtree(path_pictures_prob)
        os.makedirs(path_pictures_prob)

    for i in range(1, 12):
        for j in range(1, 12):
            if i > j:
                meta_1 = f'meta_{i}'
                meta_2 = f'meta_{j}'
                fig = coverage(df_prob_meta_red, meta_1, meta_2)
                # save figure
                fig.write_image(file = path_pictures_prob + f'prob_meta_{i}_meta_{j}.webp')

    fig = go.Figure()
    fig = make_subplots(rows=12, cols=12, vertical_spacing=0.01, horizontal_spacing=0.01)



    df_model_meta_metr = df_model_meta[['accuracy', 'precision', 'recall', 'roc_auc_score', 'geometric_mean_score', 'matthews_corrcoef', 'f1_weighted']]

    # define colors for average probabilities
    df_model_meta_color = pd.DataFrame()
    df_model_meta_color['prob'] = df_model_meta.average_probability
    # standarize df_test.prob from 0 to 1
    df_model_meta_color['prob_norm'] = minmax_scale(df_model_meta_color['prob'], feature_range=(0,1))
    # convert df_test.prob_norm to colors
    df_model_meta_color['color'] = df_model_meta_color['prob_norm'] * (-225) +225
    # convert to int
    df_model_meta_color['color'] = df_model_meta_color['color'].astype(int)
    # convert to hex
    df_model_meta_color['color_hex'] = df_model_meta_color['color'].apply(lambda x: '#%02x%02x%02x' % (x, x, x))

    ############################################################

    # add subplots for algorithm contribution to end result

    for i in range(11):
        for j in range(11):
            if i < j:

                if df_pred_meta_cor.loc[f'meta_{i+1}', f'meta_{j+1}'] > df_prob_meta_cor.loc[f'meta_{i+1}', f'meta_{j+1}']:
                    color='#9970ab'
                else:
                    color='#2ca25f'

                fig.add_trace(go.Indicator(
                mode = 'gauge+number+delta',
                value =  df_pred_meta_cor.loc[f'meta_{i+1}', f'meta_{j+1}']/100,
                # add percentage sign to number and decimals to number
                number = {'valueformat':'.0%', 'font': {'size': 10}},
                delta = {'reference': (df_pred_meta_cor.loc[f'meta_{i+1}', f'meta_{j+1}'] - df_prob_meta_cor.loc[f'meta_{i+1}', f'meta_{j+1}'])/100, 
                        'increasing': {'color': color, 'symbol': ''}, 
                        'decreasing': {'color': color, 'symbol': ''}, 
                        'font': {'size': 10},
                        'relative': False, 
                        'valueformat':'.0%',
                        'position': "top"},
                gauge = {
                    'axis': {
                        'range': [0.5, 1], 
                        'tickwidth': 1, 
                        'tickfont': {'size': 8}, 
                        'ticklen' : 1, 
                        'tickvals': [0.6, 0.7, 0.8, 0.9], 
                        'ticktext': [60, 70, 80, 90]},
                    'bar': {'color': "#fdbf6f"},
                    'steps' : [{
                        'range': [0.5, df_prob_meta_cor.loc[f'meta_{i+1}', f'meta_{j+1}']/100], 
                        'color': "gray"}],
                    'threshold' : {'line': {
                                    'color': color, 
                                    'width': 4}, 
                    'thickness': 1, 
                    'value': df_prob_meta_cor.loc[f'meta_{i+1}', f'meta_{j+1}']/100}},
                domain = {'row': i+1, 'column': j+1}))

    fig.update_layout(
        grid = {'rows': 12, 'columns': 12, 'pattern': "independent"})
                            
    ############################################################
    # add rows and columns names
    for i in range(11):
        fig.add_annotation(text=algo_cap[i], showarrow=False, font={"size":20, 'color':'#2E5984'}, row=1, col=i+2)
    for i in range(11):
        fig.add_annotation(text=algo_cap[i], showarrow=False, font={"size":20, 'color':'#cd5c5c'}, row=i+2, col=1)

    ############################################################

    # add subplots for metamodel comparison with color coding
    # define sub_plot size
    img_width = 500
    img_height = 500

    for i in range(11):
        for j in range(11):
            if i > j:
                img = Image.open(path_pictures_prob + f'prob_meta_{i+1}_meta_{j+1}.webp')

                # # Add invisible scatter trace.
                # # This trace is added to help the autoresize logic work.
                fig.add_trace(go.Scatter(x=[0, img_width], y=[0, img_height], mode="markers", marker_opacity=0), row=i+2, col=j+2)

                # Add image
                fig.add_layout_image(dict(source=img, x=0, y = img_height-100, sizex=img_width, sizey=img_height, xref='paper', yref='paper', 
                                        opacity=1.0), row=i+2, col=j+2)
                
                fig.update_layout(showlegend=False)
                fig.update_xaxes(showticklabels=False)
                fig.update_yaxes(showticklabels=False)

    ############################################################

    # add subplots for metrics in each algorithm

    # return min value from df_model_meta_metr
    y_min = df_model_meta_metr.min().min()
    # return max value from df_model_meta_metr
    y_max = df_model_meta_metr.max().max()

    # round down min_value to closest ten
    y_min = math.floor(y_min/10)*10
    # round up max_value to closest ten
    y_max = math.ceil(y_max/10)*10
    limit = y_max - y_min

    for i in range(11):

        fig.add_trace(go.Bar(x = df_model_meta_metr.columns, y = df_model_meta_metr.iloc[i], marker_color=df_model_meta_color.color_hex.iloc[i],
                    name = algo_cap[i]), row=i+2, col=i+2)
                    # reduce text size
        fig.add_annotation(text=f'Conf.: {df_model_meta.average_probability[i]}%', showarrow=False,
                    xref="x domain",yref="y domain", yshift =40, 
                    font={"size":12, 'color':'#000000'}, row=i+2, col=i+2)
        # update y axis range
        fig.update_yaxes(range=[y_min, y_max], row=i+2, col=i+2)
        # remove legend and x axes labels
        fig.update_layout(showlegend=False)

        # define ticklabels for x axis with only first letter uppercase and rotate x axis labels
        metrics_capital = df_model_meta_metr.columns.str.capitalize().str[0:1]
        # convert to list
        metrics_capital = metrics_capital.tolist()
        metrics_capital[3] = 'C' # convert to C for ROC_AUC_SCORE
        fig.update_xaxes(tickvals=df_model_meta_metr.columns, ticktext=metrics_capital, tickfont= {"size":10, 'color':'#000000'}, tickangle=0, row=i+2, col=i+2)
        # define ticklabels for y axis with values 70 to 90
        if limit <= 10:
            fig.update_yaxes(tickvals=[y_min], ticktext=[y_min], row=i+2, col=i+2)
        else:
            fig.update_yaxes(tickvals=[y_min, y_max-10], ticktext=[y_min, y_max-10], row=i+2, col=i+2)

    ############################################################

    fig.update_layout(showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # rdefine subplot size
    fig.update_layout(width=600, height=600, plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0))


    # show xaxis and yaxis labels
    fig.update_layout(xaxis105_showticklabels=True, xaxis118_showticklabels=True, xaxis131_showticklabels=True, xaxis14_showticklabels=True,
    xaxis144_showticklabels=True, xaxis27_showticklabels=True, xaxis40_showticklabels=True, xaxis53_showticklabels=True, xaxis66_showticklabels=True,
    xaxis79_showticklabels=True, xaxis92_showticklabels=True)

    fig.update_layout(yaxis105_showticklabels=True, yaxis118_showticklabels=True, yaxis131_showticklabels=True, yaxis14_showticklabels=True,
    yaxis144_showticklabels=True, yaxis27_showticklabels=True, yaxis40_showticklabels=True, yaxis53_showticklabels=True, yaxis66_showticklabels=True,
    yaxis79_showticklabels=True, yaxis92_showticklabels=True)

    return fig