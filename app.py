from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("coursera.csv")
df = df.fillna("")

# Combine useful columns
df["combined"] = df["course"] + " " + df["skills"] + " " + df["level"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])

# Similarity matrix
similarity = cosine_similarity(tfidf_matrix)


# Function to recommend courses
def recommend_from_input(user_profile):

    user_vec = vectorizer.transform([user_profile])

    sim_scores = cosine_similarity(user_vec, tfidf_matrix)

    top_indices = sim_scores.argsort()[0][-5:][::-1]

    recommended = df.iloc[top_indices][["course", "rating"]]

    return recommended.to_dict(orient="records")




# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Recommendation route
@app.route("/recommend", methods=["POST"])
def recommend():

    desired_skill = request.form["desired_skill"]
    current_skills = request.form["current_skills"]
    level = request.form["level"]
    duration = request.form["duration"]

    # Create user profile text
    user_profile = f"{desired_skill} {current_skills} {level} {duration}"

    courses = recommend_from_input(user_profile)

    return render_template("index.html", courses=courses)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


