import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import plotly.express as px
import plotly.graph_objects as go
from recommendation_engine import MovieRecommendationEngine
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #E50914;
        text-align: center;
        margin-bottom: 2rem;
    }
    .movie-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #E50914;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load MovieLens dataset"""
    try:
        # Load the datasets
        movies = pd.read_csv('data/movies.csv')
        ratings = pd.read_csv('data/ratings.csv')
        return movies, ratings
    except FileNotFoundError:
        st.error("Dataset not found! Please run the data preparation script first.")
        st.stop()

def main():
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎯 Settings")
    
    # Load data
    with st.spinner("Loading movie data..."):
        movies, ratings = load_data()
    
    # Initialize recommendation engine
    if 'rec_engine' not in st.session_state:
        with st.spinner("Initializing recommendation engine..."):
            st.session_state.rec_engine = MovieRecommendationEngine(movies, ratings)
    
    # Main navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🎬 Get Recommendations", "📊 Analytics", "🔍 Movie Search", "ℹ️ About"])
    
    with tab1:
        recommendation_tab(st.session_state.rec_engine, movies, ratings)
    
    with tab2:
        analytics_tab(movies, ratings)
    
    with tab3:
        search_tab(movies, ratings)
    
    with tab4:
        about_tab()

def recommendation_tab(rec_engine, movies, ratings):
    st.header("Get Movie Recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Choose Recommendation Method")
        
        method = st.selectbox(
            "Select Method:",
            ["Collaborative Filtering", "Content-Based", "Matrix Factorization", "Hybrid"]
        )
        
        st.subheader("Rate Some Movies")
        
        # Sample movies for rating
        sample_movies = movies.sample(10)['title'].tolist()
        user_ratings = {}
        
        for movie in sample_movies:
            rating = st.slider(
                f"Rate '{movie}' (0 = Haven't seen, 1-5 = Rating)",
                0, 5, 0, key=f"rating_{movie}"
            )
            if rating > 0:
                user_ratings[movie] = rating
        
        num_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
        
        if st.button("Get Recommendations", type="primary"):
            if not user_ratings:
                st.warning("Please rate at least one movie!")
            else:
                with st.spinner("Generating recommendations..."):
                    try:
                        recommendations = rec_engine.get_recommendations(
                            user_ratings, method.lower().replace(" ", "_"), num_recommendations
                        )
                        st.session_state.recommendations = recommendations
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
    
    with col2:
        if 'recommendations' in st.session_state:
            st.subheader("🎬 Recommended Movies for You")
            
            for i, (movie, score) in enumerate(st.session_state.recommendations.items(), 1):
                with st.container():
                    st.markdown(f"""
                    <div class="movie-card">
                        <h4>{i}. {movie}</h4>
                        <p><strong>Recommendation Score:</strong> {score:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

def analytics_tab(movies, ratings):
    st.header("📊 Movie Analytics Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>Total Movies</h3>
            <h2>{:,}</h2>
        </div>
        """.format(len(movies)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3>Total Ratings</h3>
            <h2>{:,}</h2>
        </div>
        """.format(len(ratings)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h3>Total Users</h3>
            <h2>{:,}</h2>
        </div>
        """.format(ratings['userId'].nunique()), unsafe_allow_html=True)
    
    with col4:
        avg_rating = ratings['rating'].mean()
        st.markdown("""
        <div class="metric-container">
            <h3>Avg Rating</h3>
            <h2>{:.1f}</h2>
        </div>
        """.format(avg_rating), unsafe_allow_html=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Distribution")
        rating_dist = ratings['rating'].value_counts().sort_index()
        fig = px.bar(x=rating_dist.index, y=rating_dist.values,
                     labels={'x': 'Rating', 'y': 'Count'},
                     title="Distribution of Ratings")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Rated Movies")
        movie_ratings = ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).round(2)
        movie_ratings.columns = ['avg_rating', 'num_ratings']
        movie_ratings = movie_ratings[movie_ratings['num_ratings'] >= 50]  # Filter for movies with at least 50 ratings
        top_movies = movie_ratings.nlargest(10, 'avg_rating')
        
        # Merge with movie titles
        top_movies_with_titles = top_movies.merge(movies, on='movieId')
        
        fig = px.bar(
            x=top_movies_with_titles['avg_rating'],
            y=top_movies_with_titles['title'].str[:30],  # Truncate long titles
            orientation='h',
            labels={'x': 'Average Rating', 'y': 'Movie'},
            title="Top 10 Movies by Average Rating (50+ ratings)"
        )
        st.plotly_chart(fig, use_container_width=True)

def search_tab(movies, ratings):
    st.header("🔍 Movie Search & Explore")
    
    search_query = st.text_input("Search for movies:", placeholder="Enter movie title...")
    
    if search_query:
        # Filter movies based on search query
        filtered_movies = movies[movies['title'].str.contains(search_query, case=False, na=False)]
        
        if not filtered_movies.empty:
            st.subheader(f"Found {len(filtered_movies)} movies:")
            
            for _, movie in filtered_movies.head(20).iterrows():  # Show top 20 results
                # Get average rating for this movie
                movie_ratings = ratings[ratings['movieId'] == movie['movieId']]['rating']
                avg_rating = movie_ratings.mean() if not movie_ratings.empty else 0
                num_ratings = len(movie_ratings)
                
                with st.expander(f"{movie['title']} ({movie.get('genres', 'Unknown')})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Average Rating:** {avg_rating:.1f}/5.0")
                        st.write(f"**Number of Ratings:** {num_ratings}")
                    with col2:
                        if 'genres' in movie:
                            st.write(f"**Genres:** {movie['genres']}")
        else:
            st.warning("No movies found matching your search.")
    
    # Genre filter
    st.subheader("Browse by Genre")
    if 'genres' in movies.columns:
        # Extract unique genres
        all_genres = set()
        for genres_str in movies['genres'].dropna():
            all_genres.update(genres_str.split('|'))
        
        selected_genre = st.selectbox("Select a genre:", sorted(list(all_genres)))
        
        if selected_genre:
            genre_movies = movies[movies['genres'].str.contains(selected_genre, na=False)]
            
            # Get ratings for genre movies
            genre_movie_ratings = ratings[ratings['movieId'].isin(genre_movies['movieId'])]
            avg_ratings = genre_movie_ratings.groupby('movieId')['rating'].agg(['mean', 'count'])
            
            # Merge and sort
            genre_movies_with_ratings = genre_movies.merge(avg_ratings, on='movieId', how='left')
            genre_movies_with_ratings = genre_movies_with_ratings.sort_values('mean', ascending=False)
            
            st.write(f"Top {selected_genre} movies:")
            for _, movie in genre_movies_with_ratings.head(10).iterrows():
                avg_rating = movie['mean'] if not pd.isna(movie['mean']) else 0
                num_ratings = movie['count'] if not pd.isna(movie['count']) else 0
                
                st.write(f"⭐ {movie['title']} - {avg_rating:.1f}/5.0 ({num_ratings} ratings)")

def about_tab():
    st.header("ℹ️ About This App")
    
    st.markdown("""
    ## 🎬 Movie Recommendation System
    
    This application uses machine learning to recommend movies based on your preferences and viewing history.
    
    ### 🔧 Features:
    - **Collaborative Filtering**: Recommendations based on similar users' preferences
    - **Content-Based Filtering**: Recommendations based on movie characteristics
    - **Matrix Factorization**: Advanced technique using SVD for dimensionality reduction
    - **Hybrid Approach**: Combines multiple methods for better recommendations
    
    ### 📊 Dataset:
    - **MovieLens Dataset**: Contains movie ratings from real users
    - **Ratings**: User ratings on a scale of 1-5 stars
    - **Movies**: Movie titles, genres, and metadata
    
    ### 🛠️ Technologies Used:
    - **Streamlit**: Web application framework
    - **Pandas & NumPy**: Data manipulation and analysis
    - **Scikit-learn**: Machine learning algorithms
    - **Plotly**: Interactive visualizations
    - **Surprise**: Collaborative filtering library
    
    ### 🎯 How to Use:
    1. Rate some movies in the recommendation tab
    2. Choose your preferred recommendation method
    3. Get personalized movie recommendations
    4. Explore analytics and search for specific movies
    
    ---
    *Built with ❤️ using Python and Streamlit*
    """)

if __name__ == "__main__":
    main()
