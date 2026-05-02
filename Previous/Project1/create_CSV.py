import pandas as pd
import os

base_path = "Classes\Spring 2025\Big Data Analytics (AAI 627)\Project\Project1\Assets\music-recommender-2026s"  

# 1. albumData2 -> CSV
album_rows = []
with open(f"{base_path}/albumData2.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) >= 2:
            album_id = parts[0]
            artist_id = parts[1] if parts[1] != "None" else None
            genre_ids = parts[2:] if len(parts) > 2 else []
            genre_str = "|".join(genre_ids) if genre_ids else ""
            album_rows.append({"AlbumID": album_id, "ArtistID": artist_id, "GenreIDs": genre_str})
pd.DataFrame(album_rows).to_csv("album_data.csv", index=False)

# 2. artistData2 -> CSV (simple list)
artists = []
with open(f"{base_path}/artistData2.txt", "r", encoding="utf-8") as f:
    for line in f:
        artists.append({"ArtistID": line.strip()})
pd.DataFrame(artists).to_csv("artist_data.csv", index=False)

# 3. genreData2 -> CSV
genres = []
with open(f"{base_path}/genreData2.txt", "r", encoding="utf-8") as f:
    for line in f:
        genres.append({"GenreID": line.strip()})
pd.DataFrame(genres).to_csv("genre_data.csv", index=False)

# 4. trackData2 -> CSV (with variable genre columns)
track_rows = []
with open(f"{base_path}/trackData2.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) >= 3:
            row = {"TrackID": parts[0], "AlbumID": parts[1] if parts[1] != "None" else None, 
                   "ArtistID": parts[2] if parts[2] != "None" else None}
            for i, g in enumerate(parts[3:], 1):
                row[f"Genre{i}"] = g if g != "None" else None
            track_rows.append(row)
pd.DataFrame(track_rows).to_csv("track_data.csv", index=False)

# 5. trainItem2 -> CSV (user, item, rating)
train_rows = []
with open(f"{base_path}/trainItem2.txt", "r", encoding="utf-8") as f:
    current_user = None
    num_ratings = 0
    for line in f:
        if "|" in line and "\t" not in line:
            current_user, num_ratings = line.strip().split("|")
            num_ratings = int(num_ratings)
        else:
            parts = line.strip().split("\t")
            if len(parts) == 2 and current_user:
                train_rows.append({"UserID": current_user, "ItemID": parts[0], "Rating": int(parts[1])})
pd.DataFrame(train_rows).to_csv("train_data.csv", index=False)

# 6. testItem2 -> CSV (user, track_id, track_rank 1-6)
test_rows = []
with open(f"{base_path}/testItem2.txt", "r", encoding="utf-8") as f:
    current_user = None
    for line in f:
        if "|" in line and len(line.strip().split("|")[0]) > 5:
            current_user, _ = line.strip().split("|")
        elif current_user and line.strip():
            test_rows.append({"UserID": current_user, "TrackID": line.strip()})
pd.DataFrame(test_rows).to_csv("test_data.csv", index=False)

print("CSVs created: album_data.csv, artist_data.csv, genre_data.csv, track_data.csv, train_data.csv, test_data.csv")