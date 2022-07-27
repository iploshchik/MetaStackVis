# import libraries
import ast
import streamlit as st

# Function to extract hyperparameters from the best performing model per algorithm
@st.experimental_memo
def best_params(df):
      # Select hyperparameters for best model in each algorithm
      df_best = df.groupby('algorithm_id').apply(lambda x: x.sort_values('overall_performance', ascending=False).iloc[0])
      # reset algorithm_nr as  index
      df_best = df_best.reset_index(drop=True)
      # keep only algorithm number, name, performance and hyperparameters
      df_best = df_best[['algorithm_id', 'algorithm_name', 'overall_performance', 'hyperparameters']]
      # rename overall_performance as performance
      df_best.rename(columns={'overall_performance': 'performance'}, inplace=True)
      return df_best

# Return hyperparameters based on algorithm name
@st.experimental_memo
def get_hyperparameters(df, algorithm_name):
    return df[df['algorithm_name'] == algorithm_name]['hyperparameters']

# return value of key in dictionary
@st.experimental_memo
def get_value(dictionary, key):
    return dictionary[key]

# convert string to dictionary
@st.experimental_memo
def string_to_dict(string):
    return ast.literal_eval(string)

