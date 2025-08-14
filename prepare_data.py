import os
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path

def download_movielens_data():
    """Download and extract the MovieLens dataset."""
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # MovieLens 100K dataset URL (smaller dataset for faster processing)
    url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_filename = "ml-latest-small.zip"
    
    print("Downloading MovieLens dataset...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Download completed. Extracting files...")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall("temp")
        
        # Move files to data directory
        temp_dir = Path("temp/ml-latest-small")
        
        # Copy required files
        files_to_copy = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]
        for file in files_to_copy:
            if (temp_dir / file).exists():
                import shutil
                shutil.move(str(temp_dir / file), str(data_dir / file))
                print(f"Moved {file} to data directory")
        
        # Clean up
        os.remove(zip_filename)
        import shutil
        shutil.rmtree("temp")
        
        print("Dataset preparation completed!")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("You can manually download the dataset from:")
        print("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
        print("Extract it and place movies.csv and ratings.csv in the 'data' folder")

def preprocess_data():
    """Preprocess the downloaded data."""
    
    data_dir = Path("data")
    
    try:
        # Load the datasets
        movies = pd.read_csv(data_dir / "movies.csv")
        ratings = pd.read_csv(data_dir / "ratings.csv")
        
        print(f"Loaded {len(movies)} movies and {len(ratings)} ratings")
        
        # Basic preprocessing
        print("Preprocessing data...")
        
        # Remove movies with very few ratings (less than 5)
        movie_counts = ratings['movieId'].value_counts()
        movies_to_keep = movie_counts[movie_counts >= 5].index
        
        ratings_filtered = ratings[ratings['movieId'].isin(movies_to_keep)]
        movies_filtered = movies[movies['movieId'].isin(movies_to_keep)]
        
        print(f"After filtering: {len(movies_filtered)} movies and {len(ratings_filtered)} ratings")
        
        # Remove users with very few ratings (less than 10)
        user_counts = ratings_filtered['userId'].value_counts()
        users_to_keep = user_counts[user_counts >= 10].index
        
        ratings_final = ratings_filtered[ratings_filtered['userId'].isin(users_to_keep)]
        
        print(f"Final dataset: {len(ratings_final)} ratings from {ratings_final['userId'].nunique()} users")
        
        # Save processed data
        movies_filtered.to_csv(data_dir / "movies.csv", index=False)
        ratings_final.to_csv(data_dir / "ratings.csv", index=False)
        
        # Display basic statistics
        print("\n=== Dataset Statistics ===")
        print(f"Total movies: {len(movies_filtered):,}")
        print(f"Total ratings: {len(ratings_final):,}")
        print(f"Total users: {ratings_final['userId'].nunique():,}")
        print(f"Average rating: {ratings_final['rating'].mean():.2f}")
        print(f"Rating distribution:")
        print(ratings_final['rating'].value_counts().sort_index())
        
        # Show genre distribution
        if 'genres' in movies_filtered.columns:
            print(f"\nGenre distribution:")
            all_genres = []
            for genres in movies_filtered['genres'].dropna():
                all_genres.extend(genres.split('|'))
            
            genre_counts = pd.Series(all_genres).value_counts()
            print(genre_counts.head(10))
        
        return True
        
    except FileNotFoundError:
        print("Data files not found. Please run download first.")
        return False
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return False

def create_sample_data():
    """Create sample data for testing if download fails."""
    
    print("Creating sample data for testing...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample movies
    sample_movies = pd.DataFrame({
        'movieId': range(1, 101),
        'title': [f"Movie {i}" for i in range(1, 101)],
        'genres': np.random.choice([
            'Action|Adventure', 'Comedy', 'Drama', 'Horror|Thriller',
            'Romance', 'Sci-Fi|Fantasy', 'Documentary', 'Animation|Children',
            'Crime|Mystery', 'Western'
        ], 100)
    })
    
    # Create sample ratings
    np.random.seed(42)
    sample_ratings = []
    for user_id in range(1, 101):  # 100 users
        # Each user rates 10-30 random movies
        num_ratings = np.random.randint(10, 31)
        movie_ids = np.random.choice(range(1, 101), size=num_ratings, replace=False)
        ratings = np.random.choice([1, 2, 3, 4, 5], size=num_ratings, p=[0.1, 0.1, 0.2, 0.4, 0.2])
        
        for movie_id, rating in zip(movie_ids, ratings):
            sample_ratings.append({
                'userId': user_id,
                'movieId': movie_id,
                'rating': float(rating),
                'timestamp': 1234567890
            })
    
    sample_ratings_df = pd.DataFrame(sample_ratings)
    
    # Save sample data
    sample_movies.to_csv(data_dir / "movies.csv", index=False)
    sample_ratings_df.to_csv(data_dir / "ratings.csv", index=False)
    
    print("Sample data created successfully!")
    print(f"Created {len(sample_movies)} movies and {len(sample_ratings_df)} ratings")

def main():
    """Main function to prepare the dataset."""
    
    print("=== MovieLens Dataset Preparation ===\n")
    
    # Check if data already exists
    data_dir = Path("data")
    if (data_dir / "movies.csv").exists() and (data_dir / "ratings.csv").exists():
        print("Dataset already exists!")
        response = input("Do you want to re-download? (y/n): ").lower().strip()
        if response != 'y':
            print("Using existing dataset.")
            return
    
    # Try to download real data
    try:
        download_movielens_data()
        success = preprocess_data()
        
        if not success:
            print("Falling back to sample data...")
            create_sample_data()
            
    except Exception as e:
        print(f"Error with real data: {e}")
        print("Creating sample data instead...")
        create_sample_data()
    
    print("\n=== Data preparation completed! ===")
    print("You can now run: streamlit run app.py")

if __name__ == "__main__":
    main()
