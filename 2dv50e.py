
import pandas as pd
import umap
import streamlit as st
import plotly.express as px
from sklearn import preprocessing

# Page layout
st.set_page_config(page_title='Visually-Assisted Performance Evaluation of Metamodels in Stacking Ensemble Learning',layout='wide')

# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your model probabilities data')
uploaded_probabilities = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"], key=0)
# st.sidebar.header('Upload your target data')
# uploaded_target = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"], key=1)
st.sidebar.header('Upload your model data')
uploaded_model = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"], key=2)


# Sidebar - Specify parameter settings
st.sidebar.header('Set Parameters for UMAP')
parameter_umap_n_neighbors = st.sidebar.number_input('Number of neighbors (n_neighbors)', 5)
parameter_umap_metric = st.sidebar.selectbox('Metric', ('euclidean', 'manhattan', 'chebyshev', 'minkowski'))
parameter_umap_min_dist = st.sidebar.number_input('Minimal distance', 0.1)
st.sidebar.write('---')

st.subheader('Dataset')
st.write('The dataset consists of the model probabilities for each model in the ensemble and the target variable.')

algos = {1:'K-Nearest Neighbor', 2:'Support Vector Machine', 3:'Gaussian Naive Bayes', 4:'Multilayer Perceptron', 5:'Logistic Regression',
        6:'Linear Discriminant Analysis', 7:'Quadratic Discriminant Analysis', 8:'Random Forest', 9:'Extra Trees', 10:'Adaptive Boosting',
        11:'Gradient Boosting'}


# Function to plot the UMAP plot
def create_UMAP_chart(df_probabilities, algo_nr):
    # Create UMAP
    umap_model = umap.UMAP(n_neighbors=parameter_umap_n_neighbors, metric=parameter_umap_metric, min_dist=parameter_umap_min_dist)
    umap_embedding = umap_model.fit_transform(df_probabilities)
    #convert umap_embedding to dataframe
    df_umap = pd.DataFrame(umap_embedding, columns=['UMAP_1', 'UMAP_2'])
    # Add model name and ID
    df_umap['algorithm_nr'] = algo_nr
    df_umap['model_id'] = df_model['model_id']
    # Add model specific metrics
    df_umap['accuracy'] = df_model['mean_test_accuracy']
    df_umap['precision'] = df_model['mean_test_precision_weighted']
    df_umap['recall'] = df_model['mean_test_recall_weighted']
    df_umap['roc_auc_score'] = df_model['mean_test_roc_auc_ovo_weighted']
    df_umap['geometric_mean_score'] = df_model['geometric_mean_score_weighted']
    df_umap['matthews_corrcoef'] = df_model['matthews_corrcoef']
    df_umap['f1_weighted'] = df_model['f1_weighted']
    df_umap['log_loss'] = df_model['log_loss']

    #match algo_nr with algos keys
    df_umap['algorithm_name'] = df_umap['algorithm_nr'].map(algos)
    # add overall performance data to umap
    df_umap['performance'] = df_model['overall_performance']
    # re-scale df.performance in scale from 0 to 1 and save as new column for better visualization
    df_umap['performance_scaled'] = preprocessing.MinMaxScaler().fit_transform(df_umap['performance'].values.reshape(-1,1))
    # create new column for text of points
    df_umap['text'] = df_umap['algorithm_name'] + '<br>' + 'Performance: ' + \
            df_umap['performance'].astype(str) + '%' + '<br>' + 'Model ID: ' + df_umap['model_id'].astype(str) + '<br>' + 'Accuracy: ' + \
            df_umap['accuracy'].astype(str) + '%' + '<br>' + 'Precision: ' + df_umap['precision'].astype(str) + '%' + '<br>' + 'Recall: ' + \
            df_umap['recall'].astype(str) + '%' + '<br>' + 'ROC AUC: ' + df_umap['roc_auc_score'].astype(str) + '<br>' + 'Geometric Mean: ' + \
            df_umap['geometric_mean_score'].astype(str) + '<br>' + 'Matthews Correlation: ' + df_umap['matthews_corrcoef'].astype(str) + '<br>' + 'F1: ' + \
            df_umap['f1_weighted'].astype(str) + '<br>' + 'Log Loss: ' + df_umap['log_loss'].astype(str)
    # create new column for color coding of points
    df_umap['color'] = df_umap['performance_scaled'].apply(lambda x: 'rgb(' + str(int(x*255)) + ','+ 
            str(int(x*255)) + ','+ str(int(x*255)) + ')')
    # df_umap['color'] = df_umap['performance_scaled'].apply(lambda x: 'rgba(255, 0, 0, ' + str(x) + ')')
    # Plot UMAP
    symbols = ['circle', 'square', 'x', 'cross', 'diamond', 'star', 'hexagram', 'triangle-right', 'triangle-left', 'triangle-down', 'triangle-up']

    fig = px.scatter(df_umap, x='UMAP_1', y='UMAP_2', color='performance', hover_name='text',
            symbol = df_umap['algorithm_name'], symbol_sequence = symbols, labels=dict(UMAP_1='', UMAP_2='', performance='Performance'),
            color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(title_text='UMAP Plot')
    fig.update_layout(showlegend=False)
    # Set marker symbol shape based on algorithm
    fig.update_traces(marker=dict(size=10, opacity=0.9, line=dict(width=1, color='Black')), selector=dict(mode='markers'))
    # Remove axis labels
    fig.update_layout(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
    # add tooltip
    fig.update_layout(hovermode='closest')
    # define size of figure
    fig.update_layout(width=600, height=550)

    st.plotly_chart(fig)



if uploaded_probabilities is not None and uploaded_model is not None:
    df_probabilities = pd.read_csv(uploaded_probabilities)
    df_model = pd.read_csv(uploaded_model)
    algo_nr = df_model.algorithm_id
    # Check if the number of rows in the probabilities dataframe is equal to the number of columns in the models dataframe
    if len(df_probabilities) == len(algo_nr):
        create_UMAP_chart(df_probabilities, algo_nr)
    else:
        st.write('The number of columns in the probabilities dataframe is not equal to the number of rows in the target dataframe.')
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        path = r'D:\github\2dv50e\Data\1. Heart Disease'
        df_probabilities = pd.read_csv(path + r'\topModelsProbabilities.csv')
        df_model = pd.read_csv(path + r'\topModels.csv')
        algo_nr = df_model.algorithm_id
        create_UMAP_chart(df_probabilities, algo_nr)