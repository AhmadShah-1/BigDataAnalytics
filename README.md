# Big Data Music Recommender

## Project Goal

This project is based on the Yahoo Music recommendation assignment in [`Assets/Class`](Assets/Class). The task is:

- Use each test user's historical ratings from the training data
- Look at 6 candidate tracks for that same user
- Predict exactly 3 tracks as `1` and 3 tracks as `0`
- Rank the 6 candidates by how likely they are to be recommended

The core challenge is that users can rate multiple entity types in a shared `ItemID` space:

- `Track`
- `Album`
- `Artist`
- `Genre`

So the problem is not just classic track-only collaborative filtering. The strongest idea in the assignment is to exploit the content hierarchy:

`Track -> Album -> Artist -> Genre(s)`

## What Is In This Repo

The CSVs in [`Assets/CSV`](Assets/CSV) are the working dataset used in your notebooks and scripts:

- `train_data.csv`: `UserID`, `ItemID`, `Rating`
- `test_data.csv`: `UserID`, `TrackID`
- `track_data.csv`: maps each track to album, artist, and up to 21 genres
- `album_data.csv`, `artist_data.csv`, `genre_data.csv`: entity reference tables

From the first rows of the CSVs:

- Training ratings are explicit scores on a `0-100` scale
- Test rows contain only candidate tracks, 6 per user
- Track metadata is sparse in places: some tracks are missing album or artist IDs
- Genre information is multi-label and often the broadest signal available

The assignment PDFs and your midterm report confirm the repo snapshot is roughly:

- `12.4M` training ratings
- `224,041` tracks
- `20,000` test users
- `120,000` test rows

## Summary Of Implementations

### 1. `Previous/Project1` - First rule-based hierarchy recommender

Main idea:

- Build a `track_to_hierarchy` lookup
- Build a `user_to_ratings` lookup
- Score each candidate track from:
  - album rating
  - artist rating
  - average of rated genres
- Use a weighted average and label the top 3 of 6 as `1`

Representative rule:

- album: `0.5`
- artist: `0.3`
- genre mean: `0.2`
- default fallback: `50`

What this did well:

- It matched the assignment structure correctly
- It used the hierarchy instead of treating tracks in isolation
- It enforced the required `3 ones / 3 zeros` output format
- It revealed an important data insight: many candidate rows have no direct user history at the album/artist/genre level

Main weakness:

- The fallback was too weak. When no direct signal existed, many rows collapsed to the same default score, which makes ranking mostly arbitrary.

Result:

- This was the right starting direction conceptually, but too dependent on sparse direct matches.

### 2. `Previous/Midterm/midterm.ipynb` - Stronger heuristic pipeline

This is the clearest and most mature version of the project.

#### Strategy 1: Weighted Hierarchical Average

Formula:

- `0.40 * album_score + 0.30 * artist_score + 0.30 * genre_mean`

Kaggle score:

- `0.759`

Takeaway:

- Album-level preference was a much stronger signal than genre-heavy ranking.

#### Strategy 2: Maximum Genre Score

Formula:

- `0.70 * genre_max + 0.20 * artist_score + 0.10 * album_score`

Kaggle score:

- `0.704`

Takeaway:

- This underperformed badly.
- It likely over-trusted a single liked genre and ignored the more specific album signal.

Why it is weaker:

- Genres are broad
- Multi-genre tracks can be noisy
- A high genre match does not imply the user likes that specific album or track context

### 3. Midterm v2 - Bayesian fallback + "Dig Deeper" sibling-track logic

This was an important improvement over the original heuristic setup.

Changes:

- Replaced flat fallback with Bayesian-smoothed global item scores
- Added album sibling inference:
  - if user did not rate the album directly
  - check whether they rated other tracks from the same album
  - use the mean sibling-track rating as a proxy album preference

Fallback tier breakdown from your report:

- Direct album rating: `28.6%`
- Sibling-track inference: `4.7%`
- Global Bayesian fallback: `59.6%`
- No album ID: `7.1%`

Scores:

- Weighted Avg + Bayesian + Dig Deeper: `0.774`
- Max Genre + Bayesian + Dig Deeper: `0.708`

Improvement over prior version:

- `0.759 -> 0.774` for the weighted hierarchy
- `+0.015` absolute gain

Takeaway:

- This was a real improvement.
- More importantly, it diagnosed the actual bottleneck: sparsity is so high that fallback quality matters more than small tweaks to direct rules.

### 4. Strategy 3 - Adaptive Weight Normalization

Idea:

- Only use hierarchy levels the user actually rated directly
- Normalize over the weights that fired
- If nothing fired, then use Bayesian fallback

Why this is smart:

- It avoids diluting a strong direct signal with generic fallback values
- Example:
  - if album is rated at `90`
  - and artist/genre are missing
  - use that album evidence strongly instead of blending it with generic `50-ish` defaults

Kaggle score:

- `0.792`

Improvement:

- `0.774 -> 0.792`
- `+0.018` absolute gain over v2 weighted hierarchy

Takeaway:

- This is one of the best purely heuristic ideas in the repo.
- It respects signal quality instead of pretending all features are equally trustworthy.

### 5. Strategy 5 - Genre max-blend + track-level cold-start weighting

Idea:

- Keep Strategy 3 as the core
- Replace pure genre mean with a blend:
  - `0.6 * max + 0.4 * mean`
- Add stronger track-level global weighting for cold-start cases

Kaggle score:

- `0.779`

Takeaway:

- This was worse than Strategy 3.
- The genre blend likely reintroduced some of the same instability seen in the max-genre approach.
- Your notebook also reports:
  - `Cold-start rows with track-level Bayesian score: 0`

Why that matters:

- That specific cold-start enhancement was not helping in practice for this test setup.
- Since test users already exist in training, true zero-history user cold start was not the main problem.

### 6. `Previous/Midterm/autoencoder.ipynb` - Learned fallback on top of heuristics

This is the strongest implementation in the repo.

Architecture:

- Keep the rule-based hierarchy pipeline
- Keep direct album/artist/genre reasoning
- Keep sibling-track album inference
- Replace the weakest fallback stage with a learned track preference estimate from an autoencoder
- Blend AE output with track popularity when needed

This is important because it does not throw away the assignment logic. It uses the model where the hand-built heuristics are weakest.

Local validation summary from the notebook:

- Best local configuration: `Autoencoder Fallback | AE Blend`
- Row accuracy: `81.70%`
- Mean hits@3: `2.451 / 3`
- Exact 3-of-3 rate: `54.93%`

Tier diagnostics for the best local configuration:

- `tier1_direct`: `0.893` row accuracy
- `tier3_autoencoder`: `0.744`
- `tier2_album_siblings`: `0.735`
- `no_album_id`: `0.682`
- `tier4_global`: `0.333` on very few rows

Kaggle score:

- `0.849`

Improvement:

- `0.792 -> 0.849`
- `+0.057` absolute gain over Strategy 3

Takeaway:

- This is the strongest evidence-backed direction in the repo.
- The best architecture is not "pure deep learning" and not "pure heuristic".
- It is a hybrid system where the model is used as a personalized fallback layer.

### 7. `Previous/HW8/HW8.ipynb` - Spark ALS experiments

This folder is useful, but it is more of a supporting experiment than your best final solution for this project.

What it includes:

- PySpark ALS matrix factorization
- Rank / iteration / data-size experiments
- A hybrid score combining:
  - artist score
  - album score
  - genre max
  - track global score
  - ALS score

Reported ALS observations:

- Best rank in the homework experiment: `rank=7`, `MSE=0.829`
- More iterations helped at first, then plateaued
- More data helped the most

Why ALS is useful here:

- It captures collaborative patterns that heuristics miss
- It can provide a latent preference signal between users and items

Why it is not yet the best repo implementation:

- The homework evaluation target was regression-style MSE, not the final `top-3-of-6` ranking task
- The sample predictions include values far above `100`, which suggests calibration/scaling issues for this dataset
- The assignment data uses mixed entity types in one `ItemID` namespace, while vanilla ALS works best when the interaction space is cleanly defined
- The implementation appears more like a class exercise and prototype than a tuned final pipeline for this specific problem

## Performance Timeline

| Stage | Method | Score |
| --- | --- | ---: |
| Midterm Part 1 | Weighted Hierarchical Average | `0.759` |
| Midterm Part 1 | Max Genre | `0.704` |
| Midterm Part 2 | Weighted Avg + Bayesian + Dig Deeper | `0.774` |
| Midterm Part 2 | Max Genre + Bayesian + Dig Deeper | `0.708` |
| Further Improvement | Adaptive Weight Normalization | `0.792` |
| Further Improvement | Autoencoder Fallback | `0.849` |

## What Worked Best

The strongest consistent patterns across your implementations are:

1. Album signal mattered the most.
2. Pure genre-heavy scoring was weaker than album-centered scoring.
3. Better fallback logic mattered more than small weight tweaks.
4. Adaptive weighting was better than mixing real signals with generic defaults.
5. A hybrid architecture beat both pure rules and pure matrix-style baselines.

## What Was Incorrectly Applied Or Could Be Improved

### 1. Flat defaults were too blunt

Using a single fallback like `50` made many candidates indistinguishable. That hurts ranking because the assignment is about relative ordering inside each group of 6.

### 2. Genre was sometimes overweighted

The weaker results from max-genre strategies suggest genre should be supporting evidence, not the primary driver.

### 3. Some cold-start logic did not match the actual bottleneck

The assignment test users are present in training. The bigger issue was not brand-new users, but sparse overlap between user history and candidate-track hierarchy.

### 4. ALS was not yet aligned to the final evaluation objective

ALS was tested mostly with MSE, while the real task is constrained ranking:

- 6 candidates per user
- pick top 3
- no need to predict perfectly calibrated ratings

That means a ranking objective or pairwise/listwise approach would be more aligned than plain regression.

### 5. The best ideas are spread across notebooks instead of one clean pipeline

Right now the repo shows good experimentation, but the strongest architecture is not yet consolidated into one final, reproducible implementation script.

## Recommended Final Architecture

If the goal is the best final recommender for this project, the strongest direction is:

### Hybrid Hierarchical Recommender

Stage 1: deterministic hierarchy features

- direct album rating
- direct artist rating
- direct genre stats
- sibling-track album inference
- global Bayesian album/artist/genre priors
- missingness flags
- genre count / variance / max / mean

Stage 2: learned fallback or learned fusion

- autoencoder track-affinity score
- optional ALS latent score if properly calibrated and validated on the ranking task

Stage 3: rank only within each 6-track candidate set

- produce a final score per candidate
- sort the 6 tracks for each user
- label top 3 as `1`

## Best Next Steps

### Recommended path

1. Promote the `Strategy 3 + autoencoder fallback` idea into a single final script or module.
2. Rebuild the pipeline around reusable feature-generation functions instead of notebook-only logic.
3. Evaluate every model with a local validation split that mirrors the real task:
   - group by user
   - sample 6 candidate tracks
   - score ranking quality, hits@3, row accuracy, exact 3-of-3
4. Keep album-centered features as the core signal.
5. Use the autoencoder as fallback or as one feature in a final blend, not as a full replacement.

### Highest-value improvements

1. Turn the heuristic features into a supervised ranking dataset.
   - For each `(user, candidate track)`, create features from your current pipeline.
   - Train a lightweight model such as XGBoost, LightGBM, or logistic regression to combine them.
   - This is likely the most natural next upgrade because your feature engineering is already strong.

2. Add confidence-aware features.
   - Whether album/artist/genre came from direct rating, sibling inference, AE fallback, or global prior
   - Number of sibling tracks used
   - Number of rated genres used
   - Global item rating count

3. Separate direct evidence from fallback evidence.
   - Do not let fallback values masquerade as equal-quality signals.
   - Preserve tier/source indicators so a model can learn trust levels.

4. Try a two-model blend.
   - Model A: hierarchy-based scorer
   - Model B: latent model score (autoencoder, maybe ALS)
   - Final score: learned or tuned blend on local ranking validation

5. Calibrate any latent-model outputs.
   - ALS and AE scores should be normalized to the same scale before blending.
   - The homework ALS outputs show why this matters.

## If I Were Choosing One Final Implementation

I would build the final version around:

- Strategy 3 adaptive normalization
- Bayesian item priors
- sibling-track "Dig Deeper" album inference
- autoencoder track fallback
- a final supervised combiner if time allows

If you want the safest final submission with the strongest evidence already in the repo, the best baseline is:

- `Autoencoder Fallback` from `Previous/Midterm/autoencoder.ipynb`

If you want the best likely final system after one more serious iteration, the best upgrade is:

- a feature-based hybrid ranker that uses your existing hierarchy features plus AE score as inputs

## Bottom Line

Your experiments show a clear progression rather than random trial-and-error:

- You started with the correct hierarchy idea
- You improved it by fixing sparsity and fallback logic
- You improved it again by respecting direct-signal strength
- You reached your best result by using a learned model only where heuristics were weakest

That is a strong architecture story. The best final recommender for this project is not a pure genre rule, not plain ALS, and not a full black-box neural model. It is a hybrid hierarchical recommender with adaptive weighting and a learned personalized fallback.
