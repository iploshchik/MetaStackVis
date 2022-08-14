# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from umap_functions import *
from coverage_function import *
from supporting_functions import *
from plotting_comparison import *
import hdbscan

st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)


# define random seed
np.random.seed(42)


def on_click_hdb():
        if st.session_state.use_settings:
                st.session_state.hdb_min_cluster_size = min_cluster_size
                st.session_state.hdb_min_samples = min_samples
                st.session_state.hdb_metric = metric


# Sidebar - Specify parameter settings
st.sidebar.subheader('Set Parameters for HDBSCAN clustering')
parameter_hdb_cluster_size = st.sidebar.number_input('Min cluster size', 3, key='hdb_min_cluster_size', on_change=on_click_hdb)
parameter_hdb_min_samples = st.sidebar.number_input('Min number of samples', 5, key='hdb_min_samples', on_change=on_click_hdb)
parameter_hdb_metrics = st.sidebar.selectbox('Metric', ['euclidean', 'manhattan', 'chebyshev'], key='hdb_metric', on_change=on_click_hdb)


# add varibales from the previous page
df_model = st.session_state.df_model
df_prob = st.session_state.df_prob

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

if not st.checkbox('Use settings for highest score', key='use_settings'):
        # get min_cluster_size and min_samples from side menu, conver to int
        min_cluster_size = int(st.session_state.hdb_min_cluster_size)
        min_samples = int(st.session_state.hdb_min_samples)
        # get metric from side menu
        metric = st.session_state.hdb_metric
else:
        # get min_cluster_size and min_samples and from top row
        min_cluster_size = int(df_cluster.iloc[0]['min_cluster_size'])
        min_samples = int(df_cluster.iloc[0]['min_samples'])
        # get metric from top row
        metric = df_cluster.iloc[0]['metric']



clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric,gen_min_span_tree=True)
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

# add varibales to session state for use in the other script
st.session_state['df_model'] = df_model
st.session_state['df_prob'] = df_prob
st.session_state['cnts'] = cnts
st.session_state['min_cluster_size'] = min_cluster_size
st.session_state['min_samples'] = min_samples
st.session_state['metric'] = metric
