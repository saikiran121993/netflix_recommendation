import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json
import numpy  as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import ast
import movieposters as mp


# def fetch_poster(movie_id):
#     response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=020b311fe0559698373a16008dc6a672&language=en-US'.format(movie_id))
#     data = response.json()
#     print(data)
#     return "https://image.tmdb.org/t/p/w500/" + data['poster_path']



def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def repair_array_bound_categories(arr):
    arr=ast.literal_eval(arr)
    
    if len(arr)==0:
        return np.nan
    
    elif len(arr)==1:
        return arr[0]
    
    else:
        return arr
    

df = pd.read_parquet('data.parquet')
cosine_sim = pd.read_parquet('cosine_sim.parquet').to_numpy()




def get_recommendations(title, cosine_sim, top_k=5):
    
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:20]
    movie_indices = [i[0] for i in sim_scores if i[0] != idx]

    recommendations  = df.iloc[movie_indices] \
        .sort_values(["imdb_votes", "imdb_score"], ascending=False)\
        .reset_index(drop=True)\
        .head(top_k)
  
    recommended_movies = []
    recommended_movies_posters = []
    for x in recommendations.title:
        try:
            movie_id = df[df['title']==x].imdb_id.unique()[0]
            recommended_movies.append(x)
            # recommended_movies_posters.append(fetch_poster(movie_id))
            recommended_movies_posters.append(mp.get_poster(id=movie_id))
        except:
            recommended_movies_posters.append('Netflix_Symbol_logo.webp')
            continue
    return recommendations.reset_index(drop=True) , recommended_movies, recommended_movies_posters





####################################################################
#streamlit
##################################################################


st.header('Netflix Movie Recommendation System ')
lottie_coding = load_lottiefile("m4.json")
c1, c2 = st.columns([1,3])
with c1:
    st.write(" ")
    st.write(" ")
    st.image('netflix_logo.png', width=180)

with c2:
    st_lottie(
        lottie_coding,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",height=220
    )



selected_genres = st.sidebar.multiselect(
    "Type or select a genres from the dropdown",
    ['crime', 'drama', 'comedy', 'fantasy', 'horror', 'thriller', 'action', 'music', 'romance', 'family', 'western',\
     'european', 'war', 'animation', 'documentation', 'history', 'scifi', 'sport', 'reality']
)


indices = pd.Series(df.index, index=df["title"])

genres_titles = df[['title', 'genres']]
genres_titles["genres"]=genres_titles["genres"].apply(repair_array_bound_categories)
genres_titles = genres_titles.explode('genres').drop_duplicates()


movie_list = genres_titles[genres_titles.genres.isin(selected_genres)]['title'].unique()

selected_movie = st.sidebar.selectbox(
    "Type or select a movie from the dropdown",
    movie_list[1:]
)

if st.sidebar.button('Show Recommendation'):
    
    recommendations, names, posters = get_recommendations(selected_movie, cosine_sim=cosine_sim, top_k=5)
    
    col1, col2 = st.columns([1,2])
    with col1:
        st.write(' ')
        st.write(' ')
        st.image(posters[0], width=150)
    with col2:
        st.header(names[0])
        st.markdown(recommendations.iloc[0]['description'])
        st.markdown("IMDB rating: " + str(recommendations.iloc[0]['imdb_score']))
        st.markdown("Release Year: "+ str(recommendations.iloc[0]['release_year']))
        st.markdown("Duration: " + str(recommendations.iloc[0]['runtime']))
        st.markdown("Genres: " + str(recommendations.iloc[0]['genres']))


        
    col1, col2 = st.columns([1,2])
    with col1:
        st.write(' ')
        st.write(' ')
        st.image(posters[1], width=150)
    with col2:
        st.header(names[1])
        st.markdown(recommendations.iloc[1]['description'])
        st.markdown("IMDB rating: " + str(recommendations.iloc[1]['imdb_score']))
        st.markdown("Release Year: "+ str(recommendations.iloc[1]['release_year']))
        st.markdown("Duration: " + str(recommendations.iloc[1]['runtime']))
        st.markdown("Genres: " + str(recommendations.iloc[1]['genres']))

        
    col1, col2 = st.columns([1,2])
    with col1:
        st.write(' ')
        st.write(' ')
        st.image(posters[2], width=150)
    with col2:
        st.header(names[2])
        st.markdown(recommendations.iloc[2]['description'])
        st.markdown("IMDB rating: " + str(recommendations.iloc[2]['imdb_score']))
        st.markdown("Release Year: "+ str(recommendations.iloc[2]['release_year']))
        st.markdown("Duration: " + str(recommendations.iloc[2]['runtime']))
        st.markdown("Genres: " + str(recommendations.iloc[2]['genres']))
    
    col1, col2 = st.columns([1,2])
    with col1:
        st.write(' ')
        st.write(' ')
        st.image(posters[3], width=150)
    with col2:
        st.header(names[3])
        st.markdown(recommendations.iloc[3]['description'])
        st.markdown("IMDB rating: " + str(recommendations.iloc[3]['imdb_score']))
        st.markdown("Release Year: "+ str(recommendations.iloc[3]['release_year']))
        st.markdown("Duration: " + str(recommendations.iloc[3]['runtime']))
        st.markdown("Genres: " + str(recommendations.iloc[3]['genres']))
    
    col1, col2 = st.columns([1,2])
    with col1:
        st.write(' ')
        st.write(' ')
        st.image(posters[4], width=150)
    with col2:
        st.header(names[4])
        st.markdown(recommendations.iloc[4]['description'])
        st.markdown("IMDB rating: " + str(recommendations.iloc[4]['imdb_score']))
        st.markdown("Release Year: "+ str(recommendations.iloc[4]['release_year']))
        st.markdown("Duration: " + str(recommendations.iloc[4]['runtime']))
        st.markdown("Genres: " + str(recommendations.iloc[4]['genres']))
    



EDA = st.checkbox('Show Netflix Exploratory Data Analysis')
if EDA :
    st.write(
        "check out this [link](https://github.com/saikiran121993/netflix_recommendation/blob/main/task.ipynb)")
