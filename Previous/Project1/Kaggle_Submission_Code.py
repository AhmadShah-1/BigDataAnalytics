import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("Loading data files...")

# Load all CSV files
album_df = pd.read_csv('Assets/CSV/album_data.csv')
artist_df = pd.read_csv('Assets/CSV/artist_data.csv')
genre_df = pd.read_csv('Assets/CSV/genre_data.csv')
track_df = pd.read_csv('Assets/CSV/track_data.csv')
train_df = pd.read_csv('Assets/CSV/train_data.csv')
test_df = pd.read_csv('Assets/CSV/test_data.csv')

print(f"Albums: {len(album_df):,} rows")
print(f"Artists: {len(artist_df):,} rows")
print(f"Genres: {len(genre_df):,} rows")
print(f"Tracks: {len(track_df):,} rows")
print(f"Training ratings: {len(train_df):,} rows")
print(f"Test items: {len(test_df):,} rows")
print(f"Test users: {test_df['UserID'].nunique():,}")


# Explore the data structure
print("TRACK DATA SAMPLE (showing hierarchy)")
print(track_df.head(10))

print("\n")
print("TRAINING DATA SAMPLE (user ratings)")
print(train_df.head(10))

print("\n")
print("TEST DATA SAMPLE (tracks to predict)")
print(test_df.head(12))

# Visualize data statistics with better zoom
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Rating distribution
axes[0, 0].hist(train_df['Rating'], bins=50, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Ratings (0-100)', fontsize=12, weight='bold')
axes[0, 0].set_xlabel('Rating')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(alpha=0.3)

# 2. Ratings per user (ZOOMED)
user_rating_counts = train_df.groupby('UserID').size()
axes[0, 1].hist(user_rating_counts, bins=100, color='salmon', edgecolor='black')
axes[0, 1].set_title('Ratings per User (Full Range)', fontsize=12, weight='bold')
axes[0, 1].set_xlabel('Number of Ratings')
axes[0, 1].set_ylabel('Number of Users')
axes[0, 1].set_xlim(0, 5000)  # Zoom to see detail
axes[0, 1].grid(alpha=0.3)

# 2b. Ratings per user (EXTREME ZOOM)
axes[0, 2].hist(user_rating_counts, bins=50, color='salmon', edgecolor='black')
axes[0, 2].set_title('Ratings per User (Zoomed: 0-500)', fontsize=12, weight='bold')
axes[0, 2].set_xlabel('Number of Ratings')
axes[0, 2].set_ylabel('Number of Users')
axes[0, 2].set_xlim(0, 500)  # Extreme zoom
axes[0, 2].grid(alpha=0.3)

# 3. Item rating frequency (ZOOMED)
item_rating_counts = train_df.groupby('ItemID').size()
axes[1, 0].hist(item_rating_counts, bins=100, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('Ratings per Item (Zoomed: 0-100)', fontsize=12, weight='bold')
axes[1, 0].set_xlabel('Number of Ratings')
axes[1, 0].set_ylabel('Number of Items')
axes[1, 0].set_xlim(0, 100)  # Zoom to see the distribution
axes[1, 0].grid(alpha=0.3)

# 4. Test users per rating count (ZOOMED)
test_users = test_df['UserID'].unique()
test_user_rating_counts = train_df[train_df['UserID'].isin(test_users)].groupby('UserID').size()
axes[1, 1].hist(test_user_rating_counts, bins=100, color='plum', edgecolor='black')
axes[1, 1].set_title('Training Ratings for Test Users (Zoomed)', fontsize=12, weight='bold')
axes[1, 1].set_xlabel('Number of Ratings')
axes[1, 1].set_ylabel('Number of Test Users')
axes[1, 1].set_xlim(0, 2000)  # Zoom to see distribution
axes[1, 1].grid(alpha=0.3)

# 5. Rating value breakdown (categorical view)
rating_bins = [0, 20, 40, 60, 80, 100]
rating_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
train_df['RatingBin'] = pd.cut(train_df['Rating'], bins=rating_bins, labels=rating_labels, include_lowest=True)
rating_counts = train_df['RatingBin'].value_counts().sort_index()

axes[1, 2].bar(rating_counts.index, rating_counts.values, color='gold', edgecolor='black')
axes[1, 2].set_title('Rating Distribution by Range', fontsize=12, weight='bold')
axes[1, 2].set_xlabel('Rating Range')
axes[1, 2].set_ylabel('Count')
axes[1, 2].tick_params(axis='x', rotation=45)
axes[1, 2].grid(alpha=0.3, axis='y')

# Add count labels on bars
for i, (label, count) in enumerate(zip(rating_counts.index, rating_counts.values)):
    axes[1, 2].text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Print detailed statistics
print(f"\n{'='*60}")
print("RATING STATISTICS")
print(f"{'='*60}")
print(f"Rating Statistics:")
print(f"  Mean: {train_df['Rating'].mean():.2f}")
print(f"  Median: {train_df['Rating'].median():.2f}")
print(f"  Std Dev: {train_df['Rating'].std():.2f}")
print(f"  Min: {train_df['Rating'].min()}")
print(f"  Max: {train_df['Rating'].max()}")

print(f"\nUser Rating Statistics:")
print(f"  Mean ratings per user: {user_rating_counts.mean():.1f}")
print(f"  Median ratings per user: {user_rating_counts.median():.1f}")
print(f"  Max ratings by a user: {user_rating_counts.max():,}")
print(f"  Users with >1000 ratings: {(user_rating_counts > 1000).sum():,}")

print(f"\nItem Rating Statistics:")
print(f"  Mean ratings per item: {item_rating_counts.mean():.1f}")
print(f"  Median ratings per item: {item_rating_counts.median():.1f}")
print(f"  Max ratings for an item: {item_rating_counts.max():,}")
print(f"  Items with >50 ratings: {(item_rating_counts > 50).sum():,}")

print(f"\nTest User Statistics:")
print(f"  Test users: {len(test_user_rating_counts):,}")
print(f"  Mean training ratings: {test_user_rating_counts.mean():.1f}")
print(f"  Median training ratings: {test_user_rating_counts.median():.1f}")
print(f"  Test users with <10 ratings: {(test_user_rating_counts < 10).sum():,}")


'''
Top-Left (Rating Distribution): Shows they give mostly 0-10 or 90-100 ratings. This suggests strong likes/dislikes.

Top-Middle & Top-Right (User Activity): Most users rate very few items (<100), but some users rate thousands.

Bottom-Left (Item Popularity): Most items get very few ratings (< 10), meaning most songs/albums/artists are niche. Only a small fraction are popular.

Bottom-Middle (Test Users): Test users also have weird historis, so its likely prof gave us 1 test case per some user subset

Bottom-Right (Rating Ranges): Shows which rating values are most common
'''

print("Building track hierarchy lookup...")

# Create track to hierarchy mapping
# TrackID -> (AlbumID, ArtistID, [GenreIDs])
track_to_hierarchy = {}

for _, row in tqdm(track_df.iterrows(), total=len(track_df), desc="Processing tracks"):
    track_id = str(int(row['TrackID']))  # Convert to int first, then string
    album_id = str(int(row['AlbumID'])) if pd.notna(row['AlbumID']) else None
    artist_id = str(int(row['ArtistID'])) if pd.notna(row['ArtistID']) else None
    
    # Collect all genre columns that have values
    genre_ids = []
    for col in track_df.columns:
        if col.startswith('Genre') and pd.notna(row[col]):
            genre_ids.append(str(int(row[col])))
    
    track_to_hierarchy[track_id] = {
        'album': album_id,
        'artist': artist_id,
        'genres': genre_ids
    }

print(f"Created hierarchy for {len(track_to_hierarchy):,} tracks")

# Show example
example_track = '20'
if example_track in track_to_hierarchy:
    print(f"\nExample - Track {example_track}:")
    print(f"  Album: {track_to_hierarchy[example_track]['album']}")
    print(f"  Artist: {track_to_hierarchy[example_track]['artist']}")
    print(f"  Genres: {track_to_hierarchy[example_track]['genres']}")
else:
    print(f"\nTrack {example_track} not found. Available track IDs (first 5): {list(track_to_hierarchy.keys())[:5]}")


print("Building user rating lookup...")

# Create user to ratings mapping
# UserID -> {ItemID: Rating}
user_to_ratings = defaultdict(dict)

for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing ratings"):
    user_id = str(row['UserID'])
    item_id = str(row['ItemID'])
    rating = row['Rating']
    
    user_to_ratings[user_id][item_id] = rating

print(f"Created rating lookup for {len(user_to_ratings):,} users")


# Show example
example_user = '199808'
print(f"\nExample - User {example_user}:")
print(f"  Total ratings: {len(user_to_ratings[example_user])}")
print(f"  Sample ratings: {dict(list(user_to_ratings[example_user].items())[:5])}")

# Visualize hierarchy for a sample test user
sample_user = str(test_df.iloc[0]['UserID'])
sample_tracks = test_df[test_df['UserID'] == int(sample_user)]['TrackID'].astype(str).tolist()

print(f"Visualizing hierarchy for User {sample_user}")
print("="*80)

viz_data = []
for track_id in sample_tracks:
    hierarchy = track_to_hierarchy.get(track_id, {})
    album_id = hierarchy.get('album')
    artist_id = hierarchy.get('artist')
    genre_ids = hierarchy.get('genres', [])
    
    # Get user ratings for these items
    user_ratings = user_to_ratings[sample_user]
    track_rating = user_ratings.get(track_id, None)
    album_rating = user_ratings.get(album_id, None) if album_id else None
    artist_rating = user_ratings.get(artist_id, None) if artist_id else None
    genre_ratings = [user_ratings.get(g, None) for g in genre_ids]
    
    viz_data.append({
        'TrackID': track_id,
        'AlbumID': album_id,
        'Album_Rating': album_rating,
        'ArtistID': artist_id,
        'Artist_Rating': artist_rating,
        'GenreIDs': ', '.join(genre_ids[:3]) if genre_ids else 'None',
        'Genre_Ratings': ', '.join([str(r) if r else '-' for r in genre_ratings[:3]])
    })

viz_df = pd.DataFrame(viz_data)
print(viz_df.to_string(index=False))


def compute_track_score(user_id, track_id, track_to_hierarchy, user_to_ratings, 
                        weights={'album': 0.5, 'artist': 0.3, 'genre': 0.2},
                        default_rating=50):
    """    
    Args:
        user_id: User ID string
        track_id: Track ID string
        track_to_hierarchy: Dictionary mapping track IDs to hierarchy info
        user_to_ratings: Dictionary mapping user IDs to their ratings
        weights: Dictionary with weights for album, artist, genre
        default_rating: Default rating when no history exists
    
    Returns:
        score: Float score for the track
    """
    # Get hierarchy (album, artist, genres) and user ratings (album, artist, genres)
    hierarchy = track_to_hierarchy.get(track_id, {})
    user_ratings = user_to_ratings.get(user_id, {})
    
    # Extract hierarchy info
    album_id = hierarchy.get('album')
    artist_id = hierarchy.get('artist')
    genre_ids = hierarchy.get('genres', [])
    
    # Check if user has rated the album, artist, or genres
    album_rating = user_ratings.get(album_id) if album_id else None
    artist_rating = user_ratings.get(artist_id) if artist_id else None
    genre_ratings = [user_ratings.get(g) for g in genre_ids if g in user_ratings]
    

    '''
    Here we track the total score and total weight
    For example if a user rated a album:
    score += 0.5 * 70
    total_weight += 0.5

    If he has genre ratings (and in the case we have several):
    1. Average
    2. Apply weight
    3. Add to total score
    '''


    # Calculate weighted score
    score = 0
    total_weight = 0
    
    # Album score
    if album_rating is not None:
        score += weights['album'] * album_rating
        total_weight += weights['album']
    
    # Artist score
    if artist_rating is not None:
        score += weights['artist'] * artist_rating
        total_weight += weights['artist']
    
    # Genre score (average of all genre ratings)
    if genre_ratings:
        avg_genre_rating = np.mean(genre_ratings)
        score += weights['genre'] * avg_genre_rating
        total_weight += weights['genre']
    
    '''
    For the final score, we normalize it.
    In the event that maybe the user provided values for the album, but not artist, we want to reflect that effect in the score.
    '''
    # If we have some ratings, normalize by actual weight used
    if total_weight > 0:
        score = score / total_weight
    else:
        # No ratings found for any hierarchy level - use default
        score = default_rating
    
    return score


# Test the scoring function
print("Testing scoring function on sample user...")
print("="*80)

sample_user = str(test_df.iloc[0]['UserID'])
sample_tracks = test_df[test_df['UserID'] == int(sample_user)]['TrackID'].astype(str).tolist()

test_scores = []
for track_id in sample_tracks:
    score = compute_track_score(sample_user, track_id, track_to_hierarchy, user_to_ratings)
    test_scores.append({'TrackID': track_id, 'Score': score})

test_scores_df = pd.DataFrame(test_scores).sort_values('Score', ascending=False)
print(f"\nScores for User {sample_user}:")
print(test_scores_df.to_string(index=False))
print(f"\nTop 3 tracks (predict 1): {test_scores_df.head(3)['TrackID'].tolist()}")
print(f"Bottom 3 tracks (predict 0): {test_scores_df.tail(3)['TrackID'].tolist()}")


print("Generating predictions for all test users...")
print("="*80)

predictions = []

# Group test data by user
test_grouped = test_df.groupby('UserID')

for user_id, group in tqdm(test_grouped, desc="Processing users"):
    user_id_str = str(user_id)
    track_ids = group['TrackID'].astype(str).tolist()
    
    # Score each track
    scores = []
    for track_id in track_ids:
        score = compute_track_score(user_id_str, track_id, track_to_hierarchy, user_to_ratings)
        scores.append((track_id, score))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Top 3 get 1, bottom 3 get 0
    for i, (track_id, score) in enumerate(scores):
        predictor = 1 if i < 3 else 0
        predictions.append({
            'TrackID': f"{user_id}_{track_id}",
            'Predictor': predictor
        })

predictions_df = pd.DataFrame(predictions)

print(f"✓ Generated {len(predictions_df):,} predictions")
print(f"✓ Predictions with 1: {(predictions_df['Predictor'] == 1).sum():,}")
print(f"✓ Predictions with 0: {(predictions_df['Predictor'] == 0).sum():,}")
print("\nSample predictions:")
print(predictions_df.head(12))


# Validate the submission format
print("Validating submission...")
print("="*80)

# Check counts
num_users = test_df['UserID'].nunique()
expected_predictions = num_users * 6
actual_predictions = len(predictions_df)

print(f"Expected predictions: {expected_predictions:,}")
print(f"Actual predictions: {actual_predictions:,}")
print(f"Match: {'✓' if expected_predictions == actual_predictions else '✗'}")

# Extract UserID from TrackID column for faster grouping
predictions_df['UserID_extracted'] = predictions_df['TrackID'].str.split('_').str[0].astype(int)

# Group by user and validate - MUCH FASTER
user_validation = predictions_df.groupby('UserID_extracted')['Predictor'].agg(['sum', 'count'])
user_validation.columns = ['num_ones', 'total']
user_validation['num_zeros'] = user_validation['total'] - user_validation['num_ones']

# Check for errors
invalid_users = user_validation[(user_validation['num_ones'] != 3) | (user_validation['num_zeros'] != 3)]

validation_errors = len(invalid_users)

if validation_errors == 0:
    print("✓ All users have exactly 3 ones and 3 zeros")
else:
    print(f"✗ {validation_errors} users have incorrect prediction counts")
    print("\nFirst 5 errors:")
    print(invalid_users.head())

# Drop the temporary column before saving
predictions_df = predictions_df.drop('UserID_extracted', axis=1)

# Save submission file
output_file = 'submission.csv'
predictions_df.to_csv(output_file, index=False)
print(f"\n✓ Submission saved to: {output_file}")

# Display summary statistics
print("\n")
print("SUBMISSION SUMMARY")
print(f"Total users: {num_users:,}")
print(f"Total predictions: {len(predictions_df):,}")
print(f"Recommendations (1): {(predictions_df['Predictor'] == 1).sum():,} ({(predictions_df['Predictor'] == 1).sum()/len(predictions_df)*100:.1f}%)")
print(f"Non-recommendations (0): {(predictions_df['Predictor'] == 0).sum():,} ({(predictions_df['Predictor'] == 0).sum()/len(predictions_df)*100:.1f}%)")

