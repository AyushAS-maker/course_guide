import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
df = pd.read_csv("Coursera.csv")
df = df.fillna("")
print(df.shape)
print(df.columns)
df["combined"] = df["course"] + " " + df["skills"] + " " + df["level"] + " " + df["certificatetype"]
print(df["combined"].head())
vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(df["combined"])
print(tfidf_matrix.shape)

similarity = cosine_similarity(tfidf_matrix)

def recommend_from_input(user_input):

    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix)

    scores = similarity[0]
    indices = scores.argsort()[-5:][::-1]

    return df.iloc[indices]["course"].drop_duplicates().values
courses = recommend_from_input("AI Python Beginner")

print("Recommended Courses:\n")

for course in courses:
    print("-", course)