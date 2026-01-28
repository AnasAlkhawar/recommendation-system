# -*- coding: utf-8 -*-
"""
Music Recommendation System (cleaned for GitHub/local runs)
"""

# =========================
# Imports
# =========================
import os
import re
import ast
import random
import logging
import warnings
from math import sqrt
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine, euclidean, hamming, cdist

from nltk.corpus import stopwords
from fuzzywuzzy import fuzz, process

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import util

from lightfm import LightFM, cross_validation
from lightfm.evaluation import precision_at_k, auc_score

from yellowbrick.target import FeatureCorrelation

# =========================
# Settings
# =========================
warnings.filterwarnings("ignore")
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

# =========================
# File Paths (LOCAL)
# =========================
# Your repo should have:
# data/
#   data.csv
#   data_by_genres.csv
#   data_by_year.csv
#   data_by_artist.csv
#   spotify_dataset.csv   (if you want to run the playlist section)

DATA_DIR = "data"
DATA_CSV = os.path.join(DATA_DIR, "data.csv")
GENRE_CSV = os.path.join(DATA_DIR, "data_by_genres.csv")
YEAR_CSV = os.path.join(DATA_DIR, "data_by_year.csv")
ARTIST_CSV = os.path.join(DATA_DIR, "data_by_artist.csv")
SPOTIFY_PLAYLIST_CSV = os.path.join(DATA_DIR, "spotify_dataset.csv")

# =========================
# Load Datasets
# =========================
data = pd.read_csv(DATA_CSV)
genre_data = pd.read_csv(GENRE_CSV)
year_data = pd.read_csv(YEAR_CSV)
artist_data = pd.read_csv(ARTIST_CSV)

print(data.info())
print(genre_data.info())
print(artist_data.info())
print(year_data.info())

# =========================
# Cleaning / Preprocessing
# =========================
data["artists"] = data["artists"].map(lambda x: x.lstrip("[").rstrip("]"))
data["artists"] = data["artists"].map(lambda x: x[1:-1])

# =========================
# Feature Correlation
# =========================
feature_names = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence",
    "duration_ms", "explicit", "key", "mode", "year"
]

X, y = data[feature_names], data["popularity"]
features = np.array(feature_names)

visualizer = FeatureCorrelation(labels=features)
plt.rcParams["figure.figsize"] = (10, 10)
visualizer.fit(X, y)
visualizer.show()

# =========================
# EDA
# =========================
def get_decade(year):
    period_start = int(year / 10) * 10
    decade = "{} s".format(period_start)
    return decade

data["decade"] = data["year"].apply(get_decade)

sns.set_style("whitegrid")
sns.set(rc={"figure.figsize": (10, 6)})
sns.displot(data, x="decade", height=6, aspect=2)

sound_features = ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "valence"]
fig = px.line(year_data, x="year", y=sound_features)
fig.show()

top10_genres = genre_data.nlargest(10, "popularity")
fig = px.bar(top10_genres, x="genres", y=["valence", "energy", "danceability", "acousticness"], barmode="group")
fig.show()

fig = px.line(year_data, x="year", y="loudness", title="Loudness")
fig.show()

top10_popular_artists = artist_data.nlargest(10, "popularity")
top10_most_song_produced_artists = artist_data.nlargest(10, "count")

print("Top 10 Artists that produced most songs:")
print(top10_most_song_produced_artists[["count", "artists"]].sort_values("count", ascending=False))

print("Top 10 Artists that had most popularity score:")
print(top10_popular_artists[["popularity", "artists"]].sort_values("popularity", ascending=False))

# =========================
# Clustering (Genres)
# =========================
cluster_pipeline = Pipeline([("scaler", StandardScaler()), ("kmeans", KMeans(n_clusters=12))])
X_genre = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X_genre)
genre_data["cluster"] = cluster_pipeline.predict(X_genre)

tsne_params = {
    "n_components": 2,
    "perplexity": 30,
    "learning_rate": 50,
    "verbose": 1,
    "random_state": 42
}

kmeans_params = {"n_clusters": 12}

tsne_pipeline = Pipeline([("scaler", StandardScaler()), ("tsne", TSNE(**tsne_params))])
kmeans_pipeline = Pipeline([("scaler", StandardScaler()), ("kmeans", KMeans(**kmeans_params))])

genre_embedding = tsne_pipeline.fit_transform(X_genre)
kmeans_pipeline.fit(genre_embedding)
genre_data["cluster"] = kmeans_pipeline.predict(genre_embedding)

projection = pd.DataFrame(columns=["x", "y"], data=genre_embedding)
projection["genres"] = genre_data["genres"]
projection["cluster"] = genre_data["cluster"]

fig = px.scatter(
    projection, x="x", y="y", color="cluster", hover_data=["x", "y", "genres"],
    title="Clusters of genres"
)
fig.show()

# =========================
# Clustering (Songs)
# =========================
song_cluster_pipeline = Pipeline(
    [("scaler", StandardScaler()), ("kmeans", KMeans(n_clusters=25, verbose=False))],
    verbose=False
)

X_song = data.select_dtypes(np.number)
song_cluster_pipeline.fit(X_song)
song_cluster_labels = song_cluster_pipeline.predict(X_song)
data["cluster_label"] = song_cluster_labels

pca_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("PCA", PCA(
        n_components=2,
        random_state=50,
        svd_solver="auto",
        whiten=True,
        tol=0.0,
        iterated_power="auto",
        copy=True
    ))
])

song_embedding = pca_pipeline.fit_transform(X_song)

projection = pd.DataFrame(columns=["x", "y"], data=song_embedding)
projection["title"] = data["name"]
projection["cluster"] = data["cluster_label"]

fig = px.scatter(
    projection, x="x", y="y", color="cluster", hover_data=["x", "y", "title"],
    title="Clusters of songs"
)
fig.show()

# =========================
# Optional: Playlist Dataset Section
# =========================
p = 0.02

# This section requires spotify_dataset.csv inside data/
# If you don't have it / don't want it, you can comment out this whole block
if os.path.exists(SPOTIFY_PLAYLIST_CSV):
    df_playlist = pd.read_csv(
        SPOTIFY_PLAYLIST_CSV,
        on_bad_lines="skip",
        skiprows=lambda i: i > 0 and random.random() > p
    )

    df_playlist.head()

    df_playlist.columns = df_playlist.columns.str.replace('"', "")
    df_playlist.columns = df_playlist.columns.str.replace("name", "")
    df_playlist.columns = df_playlist.columns.str.replace(" ", "")
    df_playlist.columns

    df_playlist = df_playlist.groupby("artist").filter(lambda x: len(x) >= 50)
    df_playlist = df_playlist[df_playlist.groupby("user_id").artist.transform("nunique") >= 10]

    df_freq = (
        df_playlist.groupby(["user_id", "artist"])
        .agg("size")
        .reset_index()
        .rename(columns={0: "freq"})[["user_id", "artist", "freq"]]
        .sort_values(["freq"], ascending=False)
    )
    df_freq.head()

    df_artist = pd.DataFrame(df_freq["artist"].unique())
    df_artist = df_artist.reset_index()
    df_artist = df_artist.rename(columns={"index": "artist_id", 0: "artist"})
    df_artist.head()

# =========================
# Spotify API Client
# =========================
# IMPORTANT (for GitHub):
# Do NOT hardcode your client_id/client_secret in a public repo.
# Use environment variables instead:
#   set SPOTIFY_CLIENT_ID=...
#   set SPOTIFY_CLIENT_SECRET=...
client_id = os.getenv("SPOTIFY_CLIENT_ID", "PUT_YOUR_CLIENT_ID_HERE")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "PUT_YOUR_CLIENT_SECRET_HERE")

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# =========================
# Recommendation Logic (UNCHANGED)
# =========================
def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q="track: {} year: {}".format(name, year), limit=1)
    if results["tracks"]["items"] == []:
        return None

    results = results["tracks"]["items"][0]
    track_id = results["id"]
    audio_features = sp.audio_features(track_id)[0]

    song_data["name"] = [name]
    song_data["year"] = [year]
    song_data["explicit"] = [int(results["explicit"])]
    song_data["duration_ms"] = [results["duration_ms"]]
    song_data["popularity"] = [results["popularity"]]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


number_cols = [
    "valence", "year", "acousticness", "danceability", "duration_ms", "energy", "explicit",
    "instrumentalness", "key", "liveness", "loudness", "mode", "popularity", "speechiness", "tempo"
]


def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[
            (spotify_data["name"] == song["name"]) & (spotify_data["year"] == song["year"])
        ].iloc[0]
        print("Fetching song information from local dataset")
        return song_data

    except IndexError:
        print("Fetching song information from spotify dataset")
        return find_song(song["name"], song["year"])


def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print(f"Warning: {song['name']} does not exist in Spotify or in database")
            continue
        song_vectors.append(song_data[number_cols].values)

    if not song_vectors:
        print("No valid songs found in list")
        return None

    song_matrix = np.array(song_vectors)
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    flattened_dict = {}
    for d in dict_list:
        for k, v in d.items():
            if k not in flattened_dict:
                flattened_dict[k] = []
            flattened_dict[k].append(v)
    return flattened_dict


def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ["name", "year", "artists"]
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)

    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))

    distances = cdist(scaled_song_center, scaled_data, "cosine")
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs["name"].isin(song_dict["name"])]
    return rec_songs[metadata_cols].to_dict(orient="records")


# =========================
# Example Calls (UNCHANGED)
# =========================
print(recommend_songs([{"name": "Blinding Lights", "year": 2019}], data))
print(recommend_songs([{"name": "Shirt", "year": 2022}], data))
print(recommend_songs([{"name": "One and Only", "year": 2011}], data))
print(recommend_songs([{"name": "Yellow Ledbetter", "year": 2003}], data))
print(recommend_songs([{"name": "Ramana", "year": 2006}], data))
print(recommend_songs([{"name": "Se Preparó", "year": 2017}], data))
print(recommend_songs([{"name": "Red", "year": 2020}], data))
print(recommend_songs([{"name": "all mine ", "year": 2022}], data))

print(recommend_songs([
    {"name": "Como habla una mujer", "year": 2020},
    {"name": "Mi historia entre tus dedos ", "year": 1995},
    {"name": "DE CAROLINA", "year": 2022},
    {"name": "Gatúbela", "year": 2022},
    {"name": "Ojitos lindos", "year": 2022}
], data))