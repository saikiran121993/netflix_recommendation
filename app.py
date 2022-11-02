import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json
import numpy  as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests


def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=020b311fe0559698373a16008dc6a672&language=en-US'.format(movie_id))
    data = response.json()
    print(data)
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']



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

    recommendations  = df.iloc[movie_indices] \
        .sort_values(["imdb_votes", "imdb_score"], ascending=False)[
            ["title", "description", "genres", "imdb_score", 'imdb_id']
        ] \
        .reset_index(drop=True)\
        .head(top_k)
  
    recommended_movies = []
    recommended_movies_posters = []
    for x in recommendations.title:
        try:
            movie_id = df[df['title']==x].imdb_id.unique()[0]
            recommended_movies.append(x)
            recommended_movies_posters.append(fetch_poster(movie_id))
        except:
            continue
    return recommended_movies, recommended_movies_posters



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
    # recommended_movie_names = get_recommendations(selected_movie, cosine_sim=cosine_sim, top_k=10)    
    #list_of_recommended_movie = recommended_movie_names.to_list()
   # st.write(recommended_movie_names[['title', 'description']])
    # st.write(recommended_movie_names)
    
    names, posters = get_recommendations(selected_movie, cosine_sim=cosine_sim, top_k=5)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
    
st.write('  '
         )
st.write(' ')


# EDA = st.checkbox('Show Netflix Exploratory Data Analysis')
# if EDA :
#     st.write(
#         "check out this [link](https://www.kaggle.com/code/rushikeshdane20/in-depth-analysis-of-netflix-with-plotly)")
