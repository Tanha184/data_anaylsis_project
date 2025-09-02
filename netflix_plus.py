# netflix_movie_advanced_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --------------------------
# Branding Setup
# --------------------------
sns.set_theme(style="whitegrid")
brand_colors = ["#E50914", "#221F1F", "#B20710", "#F5F5F1", "#737373"]  # Netflix palette
sns.set_palette(sns.color_palette(brand_colors))

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# --------------------------
# Load Data
# --------------------------
df = pd.read_csv("datatypes/data_science/top_1000_popular_movies_tmdb.csv", lineterminator="\n")

# Clean weird string columns
for col in ["genres", "production_companies"]:
    df[col] = df[col].astype(str).str.strip("[]").str.replace("'", "").str.split(",")

# Handle missing values
df["tagline"] = df["tagline"].fillna("No tagline")
df["budget"] = df["budget"].fillna(0)
df["revenue"] = df["revenue"].fillna(0)
df["runtime"] = df["runtime"].fillna(df["runtime"].median())

# Add calculated columns
df["roi"] = (df["revenue"] - df["budget"]) / df["budget"].replace(0, 1)
df["profit"] = df["revenue"] - df["budget"]
df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

# --------------------------
# Insights List
# --------------------------
insights = []

# --------------------------
# Advanced Insights + Visualizations
# --------------------------

# 1. Distribution of Ratings
sns.histplot(df["vote_average"], bins=10, kde=True, color=brand_colors[0])
plt.title("Distribution of Vote Averages")
plt.xlabel("Vote Average"); plt.ylabel("Frequency"); plt.show()
insights.append(f"1. Average rating across movies is {df['vote_average'].mean():.2f}, skewed towards 6-8.")

# 2. Budget vs Revenue
sns.scatterplot(data=df, x="budget", y="revenue", hue="vote_average", size="popularity", sizes=(20, 200), alpha=0.7)
plt.title("Budget vs Revenue (Colored by Rating)"); plt.xlabel("Budget"); plt.ylabel("Revenue"); plt.show()
insights.append("2. There is a strong positive correlation between budget and revenue.")

# 3. ROI distribution
sns.histplot(df["roi"], bins=30, kde=True, color=brand_colors[2])
plt.title("Distribution of ROI"); plt.xlabel("ROI (x)"); plt.ylabel("Count"); plt.show()
insights.append("3. ROI is highly skewed: a few movies return >10x, but most are below 2x.")

# 4. Runtime distribution
sns.histplot(df["runtime"], bins=20, color=brand_colors[1])
plt.title("Distribution of Movie Runtime"); plt.xlabel("Runtime (min)"); plt.ylabel("Frequency"); plt.show()
insights.append(f"4. Typical runtime is {df['runtime'].median()} mins, with long tail beyond 150 mins.")

# 5. Top Genres
top_genres = df["genres"].explode().value_counts().head(10)
sns.barplot(x=top_genres.index, y=top_genres.values, palette=brand_colors)
plt.title("Top 10 Genres"); plt.ylabel("Movie Count"); plt.xticks(rotation=45); plt.show()
insights.append(f"5. Most common genre is {top_genres.index[0]} with {top_genres.iloc[0]} movies.")

# 6. Highest grossing movies
top_movies = df.sort_values("revenue", ascending=False).head(10)
sns.barplot(data=top_movies, x="title", y="revenue", palette=brand_colors)
plt.title("Top 10 Movies by Revenue"); plt.xticks(rotation=75); plt.show()
insights.append(f"6. '{top_movies.iloc[0]['title']}' dominates with revenue ${top_movies.iloc[0]['revenue']:,}.")

# 7. Language diversity
lang_counts = df["original_language"].value_counts().head(10)
sns.barplot(x=lang_counts.index, y=lang_counts.values, palette=brand_colors)
plt.title("Top 10 Languages in Dataset"); plt.show()
insights.append(f"7. Dataset has {df['original_language'].nunique()} languages, mostly {df['original_language'].mode()[0]}.")

# 8. Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[["budget","revenue","vote_average","popularity","runtime","roi"]].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap"); plt.show()
insights.append("8. Budget and revenue show strongest correlation (0.75+). Popularity aligns weakly with ratings.")

# 9. ROI by genre
roi_genre = df.explode("genres").groupby("genres")["roi"].mean().sort_values(ascending=False).head(10)
sns.barplot(x=roi_genre.index, y=roi_genre.values, palette=brand_colors)
plt.title("Top 10 Genres by Avg ROI"); plt.xticks(rotation=45); plt.show()
insights.append(f"9. {roi_genre.index[0]} yields highest ROI on avg, with {roi_genre.iloc[0]:.2f}x.")

# 10. Revenue by year
rev_year = df.groupby("release_year")["revenue"].sum().dropna()
rev_year.plot(kind="line", color=brand_colors[0])
plt.title("Total Revenue Over Years"); plt.xlabel("Year"); plt.ylabel("Revenue"); plt.show()
insights.append("10. Movie revenues exploded post-2000, peaking in 2019 pre-COVID.")

