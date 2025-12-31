# -------------------------------
# IMPORTS
# -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Movie Success Predictor", layout="wide")
st.title("🎬 Movie Success Predictor")
st.write("Predict IMDb ratings using movie metadata")

# -------------------------------
# LOAD DATA
# -------------------------------

@st.cache_data
def load_data():
    url = "PASTE_PUBLIC_CSV_LINK_HERE"
    return pd.read_csv(url)

# @st.cache_data
# def load_data():
#     return pd.read_csv("tmdb_5000_movies.csv")

movies = load_data()

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
movies = movies[movies["budget"] > 0]

movies["runtime"].fillna(movies["runtime"].median(), inplace=True)
movies["vote_count"].fillna(0, inplace=True)
movies["release_date"].fillna("2000-01-01", inplace=True)

def count_genres(genres):
    try:
        return len(ast.literal_eval(genres))
    except:
        return 0

movies["num_genres"] = movies["genres"].apply(count_genres)
movies["log_budget"] = np.log(movies["budget"])
movies["log_vote_count"] = np.log1p(movies["vote_count"])

movies["release_year"] = pd.to_datetime(
    movies["release_date"], errors="coerce"
).dt.year
movies["movie_age"] = 2025 - movies["release_year"]

movies["is_english"] = (movies["original_language"] == "en").astype(int)

# -------------------------------
# FEATURE SELECTION
# -------------------------------
features = [
    "log_budget",
    "runtime",
    "popularity",
    "log_vote_count",
    "num_genres",
    "movie_age",
    "is_english"
]

X = movies[features]
y = movies["vote_average"]

# -------------------------------
# TRAIN MODEL (CACHED)
# -------------------------------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(X, y)

# -------------------------------
# USER INPUT (FRONTEND)
# -------------------------------
st.subheader("🎯 Enter Movie Details")

col1, col2 = st.columns(2)

with col1:
    budget = st.number_input("Budget (USD)", min_value=100000, value=50000000)
    runtime = st.number_input("Runtime (minutes)", min_value=60, value=120)
    popularity = st.slider("Popularity Score", 0.0, 100.0, 20.0)

with col2:
    vote_count = st.number_input("Vote Count", min_value=0, value=5000)
    num_genres = st.slider("Number of Genres", 1, 5, 2)
    movie_age = st.slider("Movie Age (years)", 0, 100, 5)
    is_english = st.selectbox("English Movie?", ["Yes", "No"])

input_data = pd.DataFrame([{
    "log_budget": np.log(budget),
    "runtime": runtime,
    "popularity": popularity,
    "log_vote_count": np.log1p(vote_count),
    "num_genres": num_genres,
    "movie_age": movie_age,
    "is_english": 1 if is_english == "Yes" else 0
}])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict IMDb Rating"):
    prediction = model.predict(input_data)[0]
    st.success(f"⭐ Predicted IMDb Rating: **{prediction:.2f} / 10**")

# -------------------------------
# MODEL EVALUATION
# -------------------------------
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
st.info(f"📉 Model MAE: {mae:.2f}")

# -------------------------------
# VISUALIZATION
# -------------------------------
st.subheader("📊 Predicted vs Actual")

fig, ax = plt.subplots()
ax.scatter(y_test, preds, alpha=0.5)
ax.set_xlabel("Actual Rating")
ax.set_ylabel("Predicted Rating")
st.pyplot(fig)
