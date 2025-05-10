import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('weatherHistory.csv')
    
    # Convert date column to datetime
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
    
    # Extract date components
    df['year'] = df['Formatted Date'].dt.year
    df['month'] = df['Formatted Date'].dt.month
    df['day'] = df['Formatted Date'].dt.day
    df['hour'] = df['Formatted Date'].dt.hour
    
    # Create season column
    df['season'] = df['month'].apply(lambda x: 'Winter' if x in [12,1,2] else
                                              'Spring' if x in [3,4,5] else
                                              'Summer' if x in [6,7,8] else 'Fall')
    
    # Create outfit recommendations based on temperature and precipitation
    def get_outfit(temp, precip_type, humidity):
        if temp < 5:
            outfit = 'heavy_jacket'
        elif 5 <= temp <= 15:
            outfit = 'sweater'
        else:
            outfit = 't_shirt'
        
        if precip_type == 'rain' or humidity > 80:
            outfit += '_with_umbrella'
        return outfit
    
    df['outfit'] = df.apply(
        lambda x: get_outfit(x['Temperature (C)'], x['Precip Type'], x['Humidity']), 
        axis=1
    )
    
    # Create text features for recommendation
    df['weather_features'] = df['Summary'] + ' ' + df['Precip Type'].fillna('') + ' ' + df['Daily Summary']
    
    return df

df = load_data()

# Outfit recommender class
class WeatherOutfitRecommender:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.outfit_mapping = {
            'heavy_jacket': 0,
            'heavy_jacket_with_umbrella': 1,
            'sweater': 2,
            'sweater_with_umbrella': 3,
            't_shirt': 4,
            't_shirt_with_umbrella': 5
        }
    
    def train_tfidf(self):
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['weather_features'])
    
    def recommend_outfit(self, temp, weather_desc, humidity):
        query = f"{weather_desc} {humidity} {temp}"
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)
        most_similar_idx = similarities.argmax()
        return self.df.iloc[most_similar_idx]['outfit']

# Initialize recommender
recommender = WeatherOutfitRecommender(df)
recommender.train_tfidf()

# Streamlit app
st.title("🌦️ Weather Analysis & Outfit Recommender")

# Sidebar for navigation
app_mode = st.sidebar.selectbox("Choose Mode", 
                               ["Weather Dashboard", "Outfit Recommender", "Data Exploration"])

if app_mode == "Weather Dashboard":
    st.header("Weather Data Dashboard")
    
    # Date range selector - fixed version
    min_date = df['Formatted Date'].min().date()
    max_date = df['Formatted Date'].max().date()
    
    # Create two separate date inputs for range selection
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start date", min_date)
    end_date = col2.date_input("End date", max_date)
    
    if start_date <= end_date:
        filtered_df = df[(df['Formatted Date'].dt.date >= start_date) & 
                        (df['Formatted Date'].dt.date <= end_date)]
    else:
        st.error("Error: End date must be after start date.")
        filtered_df = df
    
    # Weather metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Temperature", f"{filtered_df['Temperature (C)'].mean():.1f}°C")
    col2.metric("Max Temperature", f"{filtered_df['Temperature (C)'].max():.1f}°C")
    col3.metric("Min Temperature", f"{filtered_df['Temperature (C)'].min():.1f}°C")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Average Humidity", f"{filtered_df['Humidity'].mean():.1%}")
    col5.metric("Average Wind Speed", f"{filtered_df['Wind Speed (km/h)'].mean():.1f} km/h")
    col6.metric("Rainy Hours", f"{len(filtered_df[filtered_df['Precip Type'] == 'rain'])}")
    
    # Temperature trend
    st.subheader("Temperature Trend")
    temp_df = filtered_df.groupby(filtered_df['Formatted Date'].dt.date)['Temperature (C)'].mean()
    st.line_chart(temp_df)
    
    # Weather summary distribution
    st.subheader("Weather Summary Distribution")
    summary_counts = filtered_df['Summary'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    summary_counts.plot(kind='bar', ax=ax)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    
    # Outfit distribution
    st.subheader("Recommended Outfit Distribution")
    outfit_counts = filtered_df['outfit'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    outfit_counts.plot(kind='bar', ax=ax)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

elif app_mode == "Outfit Recommender":
    st.header("Personalized Outfit Recommendation")
    
    # User inputs
    col1, col2 = st.columns(2)
    temp = col1.slider("Current Temperature (°C)", -10.0, 40.0, 20.0)
    humidity = col2.slider("Humidity Level", 0, 100, 50)
    
    weather_options = ['Clear', 'Partly Cloudy', 'Mostly Cloudy', 'Overcast', 'Rainy', 'Foggy']
    weather_condition = st.selectbox("Weather Condition", weather_options)
    
    precip_options = ['None', 'Rain', 'Snow']
    precipitation = st.radio("Precipitation", precip_options)
    
    if st.button("Get Outfit Recommendation"):
        weather_desc = f"{weather_condition} {precipitation}"
        outfit = recommender.recommend_outfit(temp, weather_desc, humidity)
        
        st.success(f"**Recommended outfit:** {outfit.replace('_', ' ').title()}")
        
        # Display outfit image based on recommendation
        outfit_images = {
            'heavy_jacket': '🧥',
            'heavy_jacket_with_umbrella': '🧥☔',
            'sweater': '🧣',
            'sweater_with_umbrella': '🧣☔',
            't_shirt': '👕',
            't_shirt_with_umbrella': '👕☔'
        }
        
        st.subheader("Your Recommended Outfit")
        st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{outfit_images[outfit]}</h1>", 
                   unsafe_allow_html=True)
        
        # Additional tips
        st.subheader("Additional Tips")
        if "umbrella" in outfit:
            st.info("Don't forget your umbrella! It might rain or be very humid.")
        if "heavy_jacket" in outfit:
            st.info("It's cold outside! Consider wearing warm layers.")
        elif "sweater" in outfit:
            st.info("A light sweater should keep you comfortable in this weather.")
        else:
            st.info("Enjoy the warm weather! Stay hydrated.")

elif app_mode == "Data Exploration":
    st.header("Weather Data Exploration")
    
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())
    
    st.subheader("Data Statistics")
    st.write(df.describe())
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # Temperature distribution by season
    st.subheader("Temperature Distribution by Season")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='season', y='Temperature (C)', data=df, ax=ax)
    st.pyplot(fig)
    
    # Wind speed vs temperature
    st.subheader("Wind Speed vs Temperature")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Temperature (C)', y='Wind Speed (km/h)', hue='Precip Type', data=df, ax=ax)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Weather data analysis and outfit recommendation system")
