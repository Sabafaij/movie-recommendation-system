# 🎬 Movie Recommendation System

A comprehensive movie recommendation system built with Streamlit that uses multiple machine learning approaches to provide personalized movie recommendations.

## 🌟 Features

- **Multiple Recommendation Algorithms:**
  - Collaborative Filtering (User-based)
  - Content-Based Filtering
  - Matrix Factorization (SVD)
  - Hybrid Approach

- **Interactive Web Interface:**
  - Rate movies to get personalized recommendations
  - Search and explore movies
  - Analytics dashboard with visualizations
  - Genre-based browsing

- **Real Dataset:**
  - Uses MovieLens dataset with real user ratings
  - Comprehensive movie metadata including genres

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Navigate to your projects directory
cd C:\Users\FAIJ

# The project is already created in: movie-recommendation-system
cd movie-recommendation-system

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Download and prepare the MovieLens dataset
python prepare_data.py
```

This script will:
- Download the MovieLens small dataset (~1MB)
- Extract and preprocess the data
- Create `data/movies.csv` and `data/ratings.csv`
- If download fails, it creates sample data for testing

### 3. Run the Application

```bash
# Start the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 How It Works

### 1. Collaborative Filtering
- Finds users with similar movie preferences
- Recommends movies liked by similar users
- Uses cosine similarity for user matching

### 2. Content-Based Filtering
- Analyzes movie features (genres)
- Recommends movies similar to ones you've rated highly
- Uses TF-IDF vectorization for movie features

### 3. Matrix Factorization
- Uses Singular Value Decomposition (SVD)
- Reduces dimensionality to find latent factors
- Predicts ratings for unseen movies

### 4. Hybrid Approach
- Combines all three methods with weights:
  - Collaborative: 50%
  - Content-based: 30%
  - Matrix factorization: 20%

## 🎯 Usage Guide

1. **Get Recommendations:**
   - Rate some movies using the sliders
   - Choose your preferred recommendation method
   - Click "Get Recommendations"

2. **Explore Analytics:**
   - View dataset statistics
   - See rating distributions
   - Check top-rated movies

3. **Search Movies:**
   - Search for specific movies
   - Browse by genre
   - View movie details and ratings

## 📁 Project Structure

```
movie-recommendation-system/
├── app.py                 # Main Streamlit application
├── recommendation_engine.py  # Core ML algorithms
├── prepare_data.py        # Dataset preparation script
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── data/                 # Dataset directory (created after running prepare_data.py)
    ├── movies.csv
    └── ratings.csv
```

## 🛠️ Technologies Used

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Plotly
- **Dataset:** MovieLens (GroupLens Research)

## 📈 Dataset Information

The app uses the MovieLens Latest Small dataset which contains:
- ~100,000 ratings
- ~9,000 movies
- ~600 users
- Ratings from 1995-2018

## 🎮 Customization

### Adding New Recommendation Methods

1. Add your method to `MovieRecommendationEngine` class in `recommendation_engine.py`
2. Update the dropdown options in `app.py`
3. Add the method call in `get_recommendations()`

### Modifying UI

- Update the Streamlit interface in `app.py`
- Customize CSS styling in the `st.markdown()` sections
- Add new tabs or visualization components

### Using Different Datasets

1. Modify `prepare_data.py` to handle your dataset format
2. Ensure your data has columns: `userId`, `movieId`, `rating`, `title`
3. Update the recommendation engine if needed

## 🐛 Troubleshooting

### Dataset Download Issues
If the automatic download fails:
1. Manually download from: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
2. Extract to a `temp` folder
3. Copy `movies.csv` and `ratings.csv` to `data/` folder

### Memory Issues
If you encounter memory issues with large datasets:
1. Reduce the dataset size in `prepare_data.py`
2. Increase filtering thresholds (minimum ratings per user/movie)
3. Use sampling for very large datasets

### Performance Issues
For better performance:
1. Use the small MovieLens dataset
2. Implement caching for recommendations
3. Pre-compute similarity matrices

## 📚 Learning Resources

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Content-Based Filtering](https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering)
- [Matrix Factorization](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)

## 🤝 Contributing

Feel free to contribute by:
1. Adding new recommendation algorithms
2. Improving the UI/UX
3. Adding more visualization features
4. Optimizing performance

## 📄 License

This project is for educational purposes. The MovieLens dataset is provided by GroupLens Research.

---

*Built with ❤️ using Python and Streamlit*
