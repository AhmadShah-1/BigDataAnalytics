# Codex Handoff

## Project Goal

This project is a music recommendation assignment based on the Yahoo Music hierarchy-style dataset.

For each test user:

- there are exactly 6 candidate tracks
- the goal is to predict exactly 3 tracks as `1`
- and exactly 3 tracks as `0`
- the real objective is ranking the 6 candidates correctly

The training data uses a shared `ItemID` space that includes:

- tracks
- albums
- artists
- genres

So this is not a simple user-track recommender. The hierarchy matters:

- `Track -> Album -> Artist -> Genre(s)`

## Main Repo Areas

### `Assets/Class`

Purpose:

- contains the class assignment PDFs
- explains the official project framing and dataset structure

Important files:

- `Assets/Class/FinalProject.pdf`
- `Assets/Class/YahooMusic-FinalProject_Recommender_V2.pdf`

Use this folder when:

- you need the original assignment goal
- you want to confirm the 3-of-6 ranking requirement
- you want to understand the Yahoo Music hierarchy setup

### `Assets/CSV`

Purpose:

- contains the working CSV versions of the project dataset

Important files:

- `train_data.csv`: `UserID`, `ItemID`, `Rating`
- `test_data.csv`: `UserID`, `TrackID`
- `track_data.csv`: track to album / artist / genre mapping
- `album_data.csv`
- `artist_data.csv`
- `genre_data.csv`

Use this folder when:

- you need the actual data used by the notebooks
- you want to inspect entity coverage
- you want to build mappings or candidate features

### `Assets/music-recommender-2026s`

Purpose:

- original text-format version of the same dataset

Important files:

- `trainItem2.txt`
- `testItem2.txt`
- `trackData2.txt`
- `albumData2.txt`
- `artistData2.txt`
- `genreData2.txt`

Use this folder when:

- you want the raw source data before CSV conversion

### `Previous/Project1`

Purpose:

- earliest project attempt
- simple hierarchy-based recommendation rules

Important files:

- `Kaggle_Submission_Code.py`
- `exploration.ipynb`
- `submission.csv`, `submission2.csv`, `submission3.csv`, `submission4.csv`
- `submission_user_user_cf_fast.csv`

What was done here:

- basic data exploration
- hierarchy lookup creation
- user rating lookup creation
- weighted rule using album / artist / genre
- default fallback when user history was missing

Main takeaway:

- good first direction
- fallback was too weak
- too many rows collapsed to generic/default behavior

### `Previous/Midterm`

Purpose:

- strongest rule-based and hybrid experimental work

Important files:

- `midterm.ipynb`
- `autoencoder.ipynb`
- `Midterm.pdf`

What was done in `midterm.ipynb`:

- Strategy 1: weighted hierarchical average
- Strategy 2: max genre
- Bayesian global fallback
- sibling-track "Dig Deeper" album inference
- adaptive weight normalization
- later strategy variants

Reported results from this work:

- weighted hierarchy: `0.759`
- max genre: `0.704`
- weighted + Bayesian + Dig Deeper: `0.774`
- adaptive normalization: `0.792`

What was done in `autoencoder.ipynb`:

- kept hierarchy/rule-based logic
- used an autoencoder as a learned fallback
- compared local validation pipelines

Main reported result:

- autoencoder fallback Kaggle score: `0.849`

Main takeaway:

- the best direction in the repo is hybrid
- album-centered signals are strong
- better fallback logic matters a lot
- genre-heavy strategies are weaker

### `Previous/HW8`

Purpose:

- Spark ALS homework experiments
- not the best final project solution, but useful for ALS ideas

Important files:

- `HW8.ipynb`
- `HW8.pdf`
- `mypredictions.csv`

What was done:

- ALS experiments
- rank / iteration / data-size observations
- one hybrid ALS-style scoring attempt

Main takeaway:

- ALS can work as a supporting signal
- plain ALS is not naturally aligned with the mixed-entity structure of this project

### `README.md`

Purpose:

- current written summary of the project
- describes the assignment goal, experiment history, scores, and recommended direction

Use this file when:

- you want a high-level summary first

### `ALS_Test.ipynb`

Purpose:

- current multi-ALS experiment notebook
- attempts an ALS-only decomposition by training separate models for:
  - track
  - album
  - artist
  - genre

This is the main active notebook for recent experimentation.

## Current Understanding Of The Problem

The strongest signals in this project are not broad collaborative filtering alone. The best results so far suggest:

- album-level preference is very important
- artist helps
- genre is noisy
- fallback quality matters a lot because the data is sparse

The project is fundamentally a ranking problem, not a pure rating prediction problem.

Important implication:

- evaluating only MSE-style behavior is not enough
- the useful question is which method best orders 6 candidate tracks for each user

## What Has Been Done Recently In `ALS_Test.ipynb`

### Goal of `ALS_Test.ipynb`

Try an ALS-only decomposition that is cleaner than one raw mixed `ItemID` model.

Approach:

- split training interactions into:
  - `user x track`
  - `user x album`
  - `user x artist`
  - `user x genre`
- train a separate ALS model for each
- score candidate tracks by mapping them back through the hierarchy
- combine those signals into one final per-track score

### Current notebook structure

Broad sections:

1. Java / Spark setup
2. CSV loading
3. Track metadata tables
4. Entity ID lookup tables
5. Clean ALS training tables
6. ALS helper function
7. Train separate ALS models
8. Prepare candidate tables for scoring
9. Predict ALS scores for each entity space
10. Combine all entity-level predictions
11. Blend/fuse those predictions
12. Generate submission
13+. Diagnostics / comparisons / exports

### Important intermediate tables in `ALS_Test.ipynb`

#### `track_core_df`

Contains:

- `trackID`
- `albumID`
- `artistID`

Use it when:

- mapping candidate tracks to albums/artists

#### `track_genres_df`

Contains:

- one row per `trackID`, `genreID`

Use it when:

- building genre candidate scoring

#### `train_track_df`, `train_album_df`, `train_artist_df`, `train_genre_df`

Purpose:

- clean training tables for each entity-specific ALS model

#### `candidate_scores_df`

Purpose:

- combined candidate-level raw ALS predictions
- one place to inspect:
  - `track_als`
  - `album_als`
  - `artist_als`
  - `genre_mean_als`
  - `genre_max_als`

Important note:

- this table originally had duplicate `(userID, trackID)` rows because of join multiplicity
- a later dedupe step was required

#### `candidate_scores_rank_df`

Purpose:

- deduped version of the candidate score table
- safer input for later fusion methods

## ALS Experiment Findings So Far

### Multi-ALS raw score blend

Result:

- approximately `0.653`

Inference:

- the multi-ALS decomposition itself is workable
- but raw ALS outputs from different models are on incompatible scales
- direct weighted blending of raw scores is unstable

Observed issue:

- different ALS models produced very different score magnitudes
- genre especially produced very extreme values

### Rank blend

Result:

- approximately `0.646`

Inference:

- implementation was corrected later and became structurally valid
- but pure rank blending lost too much magnitude/confidence information
- fixing the scale issue alone did not help enough

### Normalized blend

Result:

- approximately `0.688`

Inference:

- better than raw-score blend and rank blend
- likely a more usable fallback than a main final recommender
- suggests that some magnitude information is useful, but raw unnormalized scores are too unstable

## Main Inferences From ALS Work

1. The ALS-only decomposition is technically valid, but weaker than the best hybrid methods in `Previous/Midterm`.
2. Track and album seem more useful than genre.
3. Genre is the noisiest ALS component.
4. Pure rank fusion is too lossy.
5. Normalized blending is more promising than raw-score blending.
6. This ALS approach may be best used as a fallback module rather than the main final recommender.

## Likely Best Next Steps

### For strongest final project performance

The best final direction is probably still:

- hierarchy-based primary model
- adaptive weighting
- Bayesian/global fallback logic
- possibly autoencoder or another learned fallback

### For ALS-specific next experiments

Best likely experiments:

- normalized blend without genre
- normalized blend with:
  - track + album only
  - track + album + artist
- compare:
  - `track_album_only`
  - `track_album_art`
  - `album_heavy`

Reason:

- previous work strongly suggests album is more predictive than genre

## Export / Reuse Status

The ALS notebook attempted to save Spark models, but saving directly with Spark ML on Windows hit a Hadoop-related issue:

- `HADOOP_HOME and hadoop.home.dir are unset`

Implication:

- Spark model `.save()` is currently blocked unless Windows Hadoop tooling is configured
- CSV-based export of fallback scores/metadata is an easier workaround

## If Another Codex Chat Continues This Work

Recommended reading order:

1. `README.md`
2. `CODEX_HANDOFF.md`
3. `Previous/Midterm/Midterm.pdf`
4. `Previous/Midterm/midterm.ipynb`
5. `Previous/Midterm/autoencoder.ipynb`
6. `ALS_Test.ipynb`

Recommended task framing:

- treat ALS work as a fallback or supporting module
- do not assume ALS-only will beat the hybrid heuristic/AE work
- focus on ranking quality, not just raw prediction scale

Recommended caution:

- notebook outputs may contain stale output from older cells if execution order changed
- always verify which cell version actually ran before trusting results

## Short Summary

This repo contains:

- the assignment materials
- converted CSV data
- early hierarchy-rule approaches
- stronger midterm heuristic and hybrid methods
- autoencoder fallback experiments
- recent multi-ALS decomposition experiments

Best recorded project direction:

- hybrid hierarchy-aware recommender with learned fallback

Current ALS status:

- normalized multi-ALS blend reached about `0.688`
- probably useful as a fallback component
- not currently competitive with the strongest hybrid results
