import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER if not already installed
nltk.download('vader_lexicon')

# Load the dataset
file_path = r"C:\Users\jaswa\OneDrive\Desktop\College scoring system\collegereview2023.csv"
df = pd.read_csv(file_path)

# Preprocessing: Drop unnamed columns, trim spaces
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df.columns = df.columns.str.strip()

# Rename columns correctly
column_mapping = {
    "college Name": "College Name",
    "Review Text": "Review Text",
    "Rating": "Rating"
}
df.rename(columns=column_mapping, inplace=True)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment_score(text):
    sentiment = sia.polarity_scores(str(text))  # Convert NaN to string
    return sentiment['compound']  # Compound score ranges from -1 to 1

# Apply sentiment analysis to reviews
df["Sentiment Score"] = df["Review Text"].apply(get_sentiment_score)

# Aggregate scores by college (Average Score)
college_scores = df.groupby("College Name")["Sentiment Score"].mean().reset_index()

# Normalize to a 100-scale for better readability
college_scores["College Score"] = ((college_scores["Sentiment Score"] + 1) / 2) * 100

# Sort colleges by score (highest to lowest)
college_scores = college_scores.sort_values(by="College Score", ascending=False)

# Display results
#print(college_scores)

#college_scores.to_csv("college_scores.csv", index=False)


# trying to predict is it positive or negative or neutral:
def categorize_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment Category"] = df["Sentiment Score"].apply(categorize_sentiment)
#worldcount   .. 
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = " ".join(df["Review Text"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
#plt.show()

import streamlit as st

st.title("📊 College Review Analysis")

college_choice = st.selectbox("Select a College", df["College Name"].unique())
college_data = df[df["College Name"] == college_choice]

st.metric("Average Sentiment Score", round(college_data["Sentiment Score"].mean(), 2))
st.metric("Average Rating", round(college_data["Rating"].mean(), 2))





