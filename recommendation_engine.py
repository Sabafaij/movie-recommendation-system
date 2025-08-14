import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class MovieRecommendationEngine:
    def __init__(self, movies_df, ratings_df):
        """
        Initialize the recommendation engine with movies and ratings data.
        
        Args:
            movies_df (pd.DataFrame): DataFrame containing movie information
            ratings_df (pd.DataFrame): DataFrame containing user ratings
        """
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.user_movie_matrix = None
        self.movie_features = None
        self.svd_model = None
        
        # Preprocess data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare and preprocess the data for different recommendation methods."""
        
        # Create user-movie matrix for collaborative filtering
        self.user_movie_matrix = self.ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Prepare content-based features
        if 'genres' in self.movies_df.columns:
            # Use genres for content-based filtering
            self.movies_df['genres_clean'] = self.movies_df['genres'].fillna('').str.replace('|', ' ')
            
            # Create TF-IDF vectors for genres
            tfidf = TfidfVectorizer(stop_words='english')
            self.movie_features = tfidf.fit_transform(self.movies_df['genres_clean'])
        
        # Prepare matrix factorization model
        self._prepare_matrix_factorization()
    
    def _prepare_matrix_factorization(self):
        """Prepare SVD model for matrix factorization."""
        try:
            # Convert to sparse matrix for efficiency
            sparse_user_movie = csr_matrix(self.user_movie_matrix.values)
            
            # Apply SVD
            self.svd_model = TruncatedSVD(n_components=50, random_state=42)
            self.svd_model.fit(sparse_user_movie)
            
        except Exception as e:
            print(f"Error in matrix factorization preparation: {e}")
            self.svd_model = None
    
    def collaborative_filtering(self, user_ratings, n_recommendations=10):
        """
        Generate recommendations using collaborative filtering.
        
        Args:
            user_ratings (dict): Dictionary of movie titles and ratings
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            dict: Dictionary of recommended movies and their scores
        """
        try:
            # Convert movie titles to movie IDs
            user_movie_ids = {}
            for title, rating in user_ratings.items():
                movie_id = self.movies_df[self.movies_df['title'] == title]['movieId']
                if not movie_id.empty:
                    user_movie_ids[movie_id.iloc[0]] = rating
            
            if not user_movie_ids:
                return {}
            
            # Create user profile vector
            user_profile = pd.Series(0.0, index=self.user_movie_matrix.columns)
            for movie_id, rating in user_movie_ids.items():
                if movie_id in user_profile.index:
                    user_profile[movie_id] = rating
            
            # Calculate cosine similarity with all users
            similarities = {}
            for user_id in self.user_movie_matrix.index:
                user_vector = self.user_movie_matrix.loc[user_id]
                # Only calculate similarity if both users have rated common movies
                common_movies = (user_profile != 0) & (user_vector != 0)
                if common_movies.sum() > 0:
                    similarity = cosine_similarity(
                        user_profile[common_movies].values.reshape(1, -1),
                        user_vector[common_movies].values.reshape(1, -1)
                    )[0][0]
                    if not np.isnan(similarity):
                        similarities[user_id] = similarity
            
            # Get top similar users
            if not similarities:
                return self._fallback_recommendations(user_movie_ids, n_recommendations)
            
            similar_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:50]
            
            # Generate recommendations based on similar users
            recommendations = {}
            for user_id, similarity in similar_users:
                user_ratings_vector = self.user_movie_matrix.loc[user_id]
                for movie_id, rating in user_ratings_vector.items():
                    if rating > 3.5 and movie_id not in user_movie_ids:  # Good rating and not already rated
                        if movie_id not in recommendations:
                            recommendations[movie_id] = 0
                        recommendations[movie_id] += similarity * rating
            
            # Convert movie IDs to titles and return top recommendations
            return self._convert_to_titles(recommendations, n_recommendations)
            
        except Exception as e:
            print(f"Error in collaborative filtering: {e}")
            return self._fallback_recommendations(user_ratings, n_recommendations)
    
    def content_based_filtering(self, user_ratings, n_recommendations=10):
        """
        Generate recommendations using content-based filtering.
        
        Args:
            user_ratings (dict): Dictionary of movie titles and ratings
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            dict: Dictionary of recommended movies and their scores
        """
        try:
            if self.movie_features is None:
                return self._fallback_recommendations(user_ratings, n_recommendations)
            
            # Get user's movie preferences
            user_movie_ids = []
            user_ratings_values = []
            
            for title, rating in user_ratings.items():
                movie_row = self.movies_df[self.movies_df['title'] == title]
                if not movie_row.empty:
                    movie_idx = movie_row.index[0]
                    user_movie_ids.append(movie_idx)
                    user_ratings_values.append(rating)
            
            if not user_movie_ids:
                return {}
            
            # Create user profile based on rated movies
            user_profile = np.zeros(self.movie_features.shape[1])
            for idx, rating in zip(user_movie_ids, user_ratings_values):
                if idx < self.movie_features.shape[0]:
                    user_profile += rating * self.movie_features[idx].toarray()[0]
            
            user_profile = user_profile / len(user_movie_ids)  # Normalize
            
            # Calculate similarity with all movies
            similarities = cosine_similarity([user_profile], self.movie_features)[0]
            
            # Get movie recommendations
            movie_scores = {}
            rated_titles = set(user_ratings.keys())
            
            for idx, score in enumerate(similarities):
                if idx < len(self.movies_df):
                    title = self.movies_df.iloc[idx]['title']
                    if title not in rated_titles and score > 0.1:  # Threshold for similarity
                        movie_scores[title] = score
            
            # Return top recommendations
            top_recommendations = dict(sorted(movie_scores.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:n_recommendations])
            
            return top_recommendations
            
        except Exception as e:
            print(f"Error in content-based filtering: {e}")
            return self._fallback_recommendations(user_ratings, n_recommendations)
    
    def matrix_factorization(self, user_ratings, n_recommendations=10):
        """
        Generate recommendations using matrix factorization (SVD).
        
        Args:
            user_ratings (dict): Dictionary of movie titles and ratings
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            dict: Dictionary of recommended movies and their scores
        """
        try:
            if self.svd_model is None:
                return self._fallback_recommendations(user_ratings, n_recommendations)
            
            # Convert movie titles to movie IDs
            user_movie_ids = {}
            for title, rating in user_ratings.items():
                movie_id = self.movies_df[self.movies_df['title'] == title]['movieId']
                if not movie_id.empty and movie_id.iloc[0] in self.user_movie_matrix.columns:
                    user_movie_ids[movie_id.iloc[0]] = rating
            
            if not user_movie_ids:
                return {}
            
            # Create a user vector
            user_vector = pd.Series(0.0, index=self.user_movie_matrix.columns)
            for movie_id, rating in user_movie_ids.items():
                user_vector[movie_id] = rating
            
            # Transform user vector using SVD
            user_vector_svd = self.svd_model.transform(user_vector.values.reshape(1, -1))
            
            # Reconstruct ratings for all movies
            reconstructed_ratings = self.svd_model.inverse_transform(user_vector_svd)[0]
            
            # Get recommendations
            recommendations = {}
            for idx, score in enumerate(reconstructed_ratings):
                movie_id = self.user_movie_matrix.columns[idx]
                if movie_id not in user_movie_ids and score > 3.0:  # Threshold for recommendations
                    recommendations[movie_id] = score
            
            return self._convert_to_titles(recommendations, n_recommendations)
            
        except Exception as e:
            print(f"Error in matrix factorization: {e}")
            return self._fallback_recommendations(user_ratings, n_recommendations)
    
    def hybrid_recommendations(self, user_ratings, n_recommendations=10):
        """
        Generate recommendations using a hybrid approach combining multiple methods.
        
        Args:
            user_ratings (dict): Dictionary of movie titles and ratings
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            dict: Dictionary of recommended movies and their scores
        """
        try:
            # Get recommendations from each method
            collab_recs = self.collaborative_filtering(user_ratings, n_recommendations * 2)
            content_recs = self.content_based_filtering(user_ratings, n_recommendations * 2)
            matrix_recs = self.matrix_factorization(user_ratings, n_recommendations * 2)
            
            # Combine recommendations with weights
            hybrid_scores = {}
            
            # Weight collaborative filtering more heavily
            for movie, score in collab_recs.items():
                hybrid_scores[movie] = hybrid_scores.get(movie, 0) + 0.5 * score
            
            # Add content-based recommendations
            for movie, score in content_recs.items():
                hybrid_scores[movie] = hybrid_scores.get(movie, 0) + 0.3 * score
            
            # Add matrix factorization recommendations
            for movie, score in matrix_recs.items():
                hybrid_scores[movie] = hybrid_scores.get(movie, 0) + 0.2 * score
            
            # Return top recommendations
            return dict(sorted(hybrid_scores.items(), 
                              key=lambda x: x[1], 
                              reverse=True)[:n_recommendations])
            
        except Exception as e:
            print(f"Error in hybrid recommendations: {e}")
            return self._fallback_recommendations(user_ratings, n_recommendations)
    
    def _convert_to_titles(self, movie_id_scores, n_recommendations):
        """Convert movie IDs to titles."""
        title_scores = {}
        for movie_id, score in movie_id_scores.items():
            movie_title = self.movies_df[self.movies_df['movieId'] == movie_id]['title']
            if not movie_title.empty:
                title_scores[movie_title.iloc[0]] = score
        
        return dict(sorted(title_scores.items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:n_recommendations])
    
    def _fallback_recommendations(self, user_ratings, n_recommendations):
        """Fallback to popular movies when other methods fail."""
        try:
            # Get popular movies based on average rating and number of ratings
            movie_stats = self.ratings_df.groupby('movieId').agg({
                'rating': ['mean', 'count']
            }).round(2)
            movie_stats.columns = ['avg_rating', 'num_ratings']
            
            # Filter for movies with at least 100 ratings and high average rating
            popular_movies = movie_stats[
                (movie_stats['num_ratings'] >= 100) & 
                (movie_stats['avg_rating'] >= 4.0)
            ].sort_values(['avg_rating', 'num_ratings'], ascending=[False, False])
            
            # Convert to titles and exclude already rated movies
            recommendations = {}
            rated_titles = set(user_ratings.keys())
            
            for movie_id in popular_movies.head(n_recommendations * 2).index:
                movie_title = self.movies_df[self.movies_df['movieId'] == movie_id]['title']
                if not movie_title.empty:
                    title = movie_title.iloc[0]
                    if title not in rated_titles:
                        recommendations[title] = popular_movies.loc[movie_id, 'avg_rating']
                        if len(recommendations) >= n_recommendations:
                            break
            
            return recommendations
            
        except Exception as e:
            print(f"Error in fallback recommendations: {e}")
            return {}
    
    def get_recommendations(self, user_ratings, method='collaborative_filtering', n_recommendations=10):
        """
        Main method to get recommendations based on specified method.
        
        Args:
            user_ratings (dict): Dictionary of movie titles and ratings
            method (str): Recommendation method to use
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            dict: Dictionary of recommended movies and their scores
        """
        if method == 'collaborative_filtering':
            return self.collaborative_filtering(user_ratings, n_recommendations)
        elif method == 'content-based':
            return self.content_based_filtering(user_ratings, n_recommendations)
        elif method == 'matrix_factorization':
            return self.matrix_factorization(user_ratings, n_recommendations)
        elif method == 'hybrid':
            return self.hybrid_recommendations(user_ratings, n_recommendations)
        else:
            return self.collaborative_filtering(user_ratings, n_recommendations)
