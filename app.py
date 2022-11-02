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




def get_recommendations(title, cosine_sim, top_k=5, years=range(1980, 2022)):
    
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:20]
    movie_indices = [i[0] for i in sim_scores if i[0] != idx]

    recommendations  = df.iloc[(movie_indices)] \
        .sort_values(["imdb_votes", "imdb_score"], ascending=False)\
        .reset_index(drop=True)\
        .drop_duplicates(subset='title')
    recommendations = recommendations[recommendations['release_year'].isin(years)]
    recommendations = recommendations.iloc[0:5].reset_index(drop=True)
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


st.subheader('Netflix Recommendation system')
st.caption('This recommendation system is using the genres and/or descriptions')
lottie_coding = load_lottiefile("m4.json")
 
c1, c2 = st.columns([1,3])
with c1:
   st.write("")
   st.write("")
   st.write("")
   st.image('netflix_logo.png', width=150)

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
     'european', 'war', 'animation', 'documentation', 'history', 'scifi', 'sport', 'reality', None]
)

production_countries = st.sidebar.multiselect(
    "Select countries from the dropdown",
    ['US', 'GB', 'JP', 'EG', 'DE', 'IN', 'DZ', 'LB', 'FR', 'CA', 'SU', \
       'IT', 'HK', 'AR', 'KW', 'PE', 'BR', 'IE', 'GH', 'BF', 'ES', 'MX',\
       'BE', 'NO', 'PS', 'TR', 'BS', 'CZ', 'AU', 'IS', 'NZ', 'CN',\
       'BG', 'MA', 'UY', 'TW', 'DK', 'KR', 'MY', 'CH', 'CL', 'NG', 'ZA',\
       'SA', 'AT', 'NL', 'SE', 'PH', 'Lebanon', 'TH', 'XX', 'AE', 'HU',\
       'ID', 'RO', 'CD', 'VE', 'IL', 'UA', 'SG', 'IR', 'JO', 'QA', 'LU',\
       'FI', 'SK', 'SY', 'GL', 'PT', 'PK', 'VN', 'PR', 'IQ', 'KH', 'GE',\
       'CU', 'PL', 'KE', 'VA', 'RU', 'RS', 'AL', 'TZ', 'TN', 'ZW', 'PY',\
       'CO', 'NP', 'GR', 'CM', 'BD', 'KG', 'LT', 'CY', 'SN', 'MW', 'MU',\
       'LK', 'AO', 'GT', 'MZ', 'AF', 'NA', 'FO']
)

release_years = st.sidebar.multiselect(
    "Select release years from the dropdown",
    df.release_year.unique()
)


indices = pd.Series(df.index, index=df["title"])

genres_titles = df[['title','production_countries', 'genres']]
genres_titles["genres"]=genres_titles["genres"].apply(repair_array_bound_categories)
genres_titles = genres_titles.explode('genres').drop_duplicates()
genres_titles["production_countries"]=genres_titles["production_countries"].apply(repair_array_bound_categories)
genres_titles = genres_titles.explode('production_countries').drop_duplicates()

movie_list = genres_titles[(genres_titles.genres.isin(selected_genres)) 
                           & (genres_titles.production_countries.isin(production_countries))
                           ]['title'].unique()

selected_movie = st.sidebar.selectbox(
    "Type or select a movie from the dropdown",
    movie_list[1:]
)

if st.sidebar.button('Show Recommendation'):
    
    recommendations, names, posters = get_recommendations(selected_movie, cosine_sim=cosine_sim, top_k=5, years=release_years)
    try:
        col1, col2 = st.columns([1,2])
        with col1:
            st.write('1. ')
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
            st.write('2. ')
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
            st.write('3. ')
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
            st.write('4. ')
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
            st.write('5. ')
            st.write(' ')
            st.image(posters[4], width=150)
        with col2:
            st.header(names[4])
            st.markdown(recommendations.iloc[4]['description'])
            st.markdown("IMDB rating: " + str(recommendations.iloc[4]['imdb_score']))
            st.markdown("Release Year: "+ str(recommendations.iloc[4]['release_year']))
            st.markdown("Duration: " + str(recommendations.iloc[4]['runtime']))
            st.markdown("Genres: " + str(recommendations.iloc[4]['genres']))
    except:
        st.write(" ")



EDA = st.checkbox('Show Netflix Exploratory Data Analysis')
if EDA :
    st.write(
        "check out this [link](https://github.com/saikiran121993/netflix_recommendation/blob/main/task.ipynb)")
