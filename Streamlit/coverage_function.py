# import libraries
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import streamlit as st


'''
Function compares 2 metamodels and plots the overall coverage, using following predefined colors:
White box: both models predict the correct class
Red box: 1st model predicts the correct class, 2nd model predicts the wrong class and combination predicts correct class
Light red box: 1st model predicts the correct class, 2nd model predicts the wrong class and combination predicts wrong class
Blue box: 1st model predicts the wrong class, while the 2nd model predicts the correct class and combination predicts correct class
Light blue box: 1st model predicts the wrong class, while the 2nd model predicts the correct class and combination predicts correct class
Yellow box: both models predict the wrong class
'''
@st.experimental_memo
def coverage(df, meta_1, meta_2):

    # save meta_1, meta_2 and target in a new dataframe
    df_temp = df[[f'{meta_1}', f'{meta_2}']].copy().reset_index(drop=True)
    # rename columns to meta_1 and meta_2
    df_temp.columns.values[0:2] = [f'meta_{i}' for i in range(1, 3)]
    df_temp['mean'] = df_temp.mean(axis=1).round(2)

    # get square root of count length and round to integer. That is to define the number of rows and columns in the plot
    n = int(np.ceil(np.sqrt(df_temp.shape[0])))

    df_n = pd.DataFrame(np.zeros((n**2 - df_temp.shape[0], 3)))
    # replace all values with nan
    df_n.iloc[:, :] = np.nan

    df_n.columns = ['meta_1', 'meta_2', 'mean']

    # concatinate df_temp and df_n
    df_temp = pd.concat([df_temp, df_n], axis=0)
    # reset index
    df_temp = df_temp.reset_index(drop=True)

    # concate values from meta_1, meta_2 and mean as a list in a new column
    df_temp['combination'] = df_temp[['meta_1', 'meta_2', 'mean']].values.tolist()

    # create new dataframe with  with n rows and n columns
    df_count = pd.DataFrame(df_temp.combination.values.reshape(n, n))

    fig = go.Figure()
    fig = make_subplots(rows=n, cols=n, vertical_spacing=0.02, horizontal_spacing=0.02, shared_xaxes='all', shared_yaxes='all')

    # define subplot size
    fig.update_layout(autosize=False, margin={'l': 0, 'r': 0, 't': 0, 'b': 50}, width=500, height=500)

    # Set axes ranges
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    for i in range(n):
        for j in range(n):
            fig.add_shape(type='circle', x0=0, y0=0, x1=1, y1=1, line=dict(width=2), row=i+1, col=j+1)
            fig.update_shapes(
                # when both models are larger then 50%, color the box white
                fillcolor='#ffffff' if (df_count.iloc[i, j][0] >= 50 and df_count.iloc[i, j][1] >= 50)
                # when meta_1 is larger then 50% and meta_2 less then 50% and combination is larger then 50%, color the box red
                else '#cd5c5c' if (df_count.iloc[i, j][0] >= 50 and df_count.iloc[i, j][1] < 50 and  df_count.iloc[i, j][2] >= 50)
                # when meta_1 is larger then 50% and meta_2 less then 50% and combination is less then 50%, color the box light red
                else '#df9797' if (df_count.iloc[i, j][0] >= 50 and df_count.iloc[i, j][1] < 50 and  df_count.iloc[i, j][2] < 50)
                # when meta_1 is less then 50% and meta_2 larger then 50% and combination is larger then 50%, color the box blue
                else '#2e5984' if (df_count.iloc[i, j][0] < 50 and df_count.iloc[i, j][1] >= 50 and  df_count.iloc[i, j][2] >= 50) 
                # when meta_1 is less then 50% and meta_2 larger then 50% and combination is less then 50%, color the box light blue
                else '#91bad6' if (df_count.iloc[i, j][0] < 50 and df_count.iloc[i, j][1] >= 50 and  df_count.iloc[i, j][2] < 50)
                # when both moels are less then 50%, color the box yellow
                else '#ffd700' if (df_count.iloc[i, j][0] < 50 and df_count.iloc[i, j][1] < 50)
                # else color the box white
                else '#ffffff', row=i+1, col=j+1)
            fig.update_shapes(
                # define circe border colors
                line=dict(color='#675c57' if (df_count.iloc[i, j][0] >= 50 and df_count.iloc[i, j][1] >= 50)
                else '#cd5c5c' if (df_count.iloc[i, j][0] >= 50 and df_count.iloc[i, j][1] < 50 and  df_count.iloc[i, j][2] >= 50)
                else '#df9797' if (df_count.iloc[i, j][0] >= 50 and df_count.iloc[i, j][1] < 50 and  df_count.iloc[i, j][2] < 50)
                else '#2e5984' if (df_count.iloc[i, j][0] < 50 and df_count.iloc[i, j][1] >= 50 and  df_count.iloc[i, j][2] >= 50) 
                else '#91bad6' if (df_count.iloc[i, j][0] < 50 and df_count.iloc[i, j][1] >= 50 and  df_count.iloc[i, j][2] < 50)
                else '#ffd700' if (df_count.iloc[i, j][0] < 50 and df_count.iloc[i, j][1] < 50) 
                else '#ffffff'), row=i+1, col=j+1)

    # remove background color
    fig.update_layout(plot_bgcolor='white')

    # remove legend and x axes labels
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
        
    return fig