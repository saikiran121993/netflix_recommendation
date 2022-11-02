import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json
import numpy  as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


df = pd.read_parquet('data.parquet')
cosine_sim = pd.read_parquet('cosine_sim.parquet').to_numpy()


indices = pd.Series(df.index, index=df["title"])
movie_list = df['title'].unique()

def get_recommendations(title, cosine_sim, top_k=5):
    
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:20]
    movie_indices = [i[0] for i in sim_scores if i[0] != idx]

    return (
        df.iloc[movie_indices]
        .sort_values(["imdb_votes", "imdb_score"], ascending=False)[
            ["title", "description", "genres", "imdb_score"]
        ]
        .reset_index(drop=True)
        .head(top_k)
    )



movie_list = df['title'].values


####################################################################
#streamlit
##################################################################


st.header('Netflix Movie Recommendation System ')
lottie_coding = load_lottiefile("m4.json")
st_lottie(
    lottie_coding,
    speed=1,
    reverse=False,
    loop=True,
    quality="low",height=220
)
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names = get_recommendations(selected_movie, cosine_sim=cosine_sim, top_k=10)    
    #list_of_recommended_movie = recommended_movie_names.to_list()
   # st.write(recommended_movie_names[['title', 'description']])
    st.write(recommended_movie_names)
    
st.write('  '
         )
st.write(' ')


# EDA = st.checkbox('Show Netflix Exploratory Data Analysis')
# if EDA :
#     st.write(
#         "check out this [link](https://www.kaggle.com/code/rushikeshdane20/in-depth-analysis-of-netflix-with-plotly)")
