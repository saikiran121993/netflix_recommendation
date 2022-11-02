import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots  import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import plotly.io as pio
pio.renderers.default = 'iframe'

import numpy as np
import pandas as pd
import os
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx 
import itertools
from itertools import combinations
import random

import ast


colors = ['#494BD3', '#E28AE2', '#F1F481', '#79DB80', '#DF5F5F',
              '#69DADE', '#C2E37D', '#E26580', '#D39F49', '#B96FE3']



def missing_values(df):
    #Let's create a dataframe, round the results, and name the columns
    df_missing_count = df.isnull().sum().reset_index().rename(columns={'index':'column_name', 0:'missing_count'})
    #Create a dataframe calculating percentage
    missing_percnt = df.isnull().sum() * 100 / df.shape[0] #df.shape[0] will give us the number of lines
    #Let's create a dataframe, round the results, and name the columns
    df_missing_percnt = pd.DataFrame(missing_percnt).round(2).reset_index().rename(columns={
                        'index':'column_name', 0:'missing_values(%)'})
   
    df = df_missing_count.merge(df_missing_percnt, how = 'inner', left_on = 'column_name', right_on = 'column_name')
    df = df.sort_values(by = 'missing_values(%)',ascending = False)
    return df



def repair_array_bound_categories_random(arr):
    arr=ast.literal_eval(arr)
    
    if len(arr)==0:
        return np.nan
    
    elif len(arr)==1:
        return arr[0]
    
    else:
        return random.choice(arr)
    
def repair_array_bound_categories(arr):
    arr=ast.literal_eval(arr)
    
    if len(arr)==0:
        return np.nan
    
    elif len(arr)==1:
        return arr[0]
    
    else:
        return arr
    
    
def cat_summary_with_graph(dataframe, col_name):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Countplot', 'Percentages'),
                        specs=[[{"type": "xy"}, {'type': 'domain'}]])

    fig.add_trace(go.Bar(y=dataframe[col_name].value_counts().values.tolist(),
                         x=[str(i) for i in dataframe[col_name].value_counts().index],
                         text=dataframe[col_name].value_counts().values.tolist(),
                         textfont=dict(size=15),
                         name=col_name,
                         textposition='auto',
                         showlegend=False,
                         marker=dict(color=colors,
                                     line=dict(color='#DBE6EC',
                                               width=1))),
                  row=1, col=1)

    fig.add_trace(go.Pie(labels=dataframe[col_name].value_counts().keys(),
                         values=dataframe[col_name].value_counts().values,
                         textfont=dict(size=20),
                         textposition='auto',
                         showlegend=False,
                         name=col_name,
                         marker=dict(colors=colors)),
                  row=1, col=2)

    fig.update_layout(title={'text': col_name,
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template='plotly_white')

    iplot(fig)
    
    
