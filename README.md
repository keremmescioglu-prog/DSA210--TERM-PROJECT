# DSA 210 - Analyzing Movie Success: What Makes a Movie Profitable?

## Project Description
This project investigates which factors — such as budget, genre, cast, release timing, and audience ratings — are most strongly associated with a movie's commercial success. Using the TMDB 5000 Movies dataset enriched with cast/crew credits data, I analyze patterns in profitability and test hypotheses about what drives box office performance.

## Data Sources
- **TMDB 5000 Movies Dataset** (Primary): ~5,000 movies with budget, revenue, genres, release date, ratings, popularity, and runtime. [Kaggle Link](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **TMDB Credits Dataset** (Enrichment): Cast and crew details for each movie, enabling analysis of director/actor effects on profitability.

## Project Structure
```
├── data/
│   ├── tmdb_5000_movies.csv      # Raw movie data
│   ├── tmdb_5000_credits.csv     # Raw credits data
│   └── movies_cleaned.csv        # Cleaned & merged dataset
├── plots/                        # Generated visualizations
├── analysis.py                   # EDA & hypothesis testing code
├── requirements.txt              # Python dependencies
└── README.md
```

## Key Findings (Milestone 1)

### EDA Highlights
- **3,196 movies** analyzed after cleaning (removed zero budget/revenue entries)
- **75.3%** of movies are profitable, with an average ROI of 3.02x
- Budget and revenue have a strong positive correlation (Spearman r = 0.69)
- Drama and Comedy are the most common genres; Horror and Animation show highest median ROI

### Hypothesis Tests
| # | Hypothesis | Test | Result |
|---|-----------|------|--------|
| 1 | High-budget movies have higher ROI | Mann-Whitney U | Significant (p < 0.001) — but low-budget movies actually have *higher* median ROI |
| 2 | Summer releases are more profitable | Mann-Whitney U | Significant (p < 0.001) — summer movies have higher ROI |
| 3 | Higher ratings correlate with higher ROI | Pearson Correlation | Significant (r = 0.24, p < 0.001) |
| 4 | Action movies earn more revenue than Drama | Mann-Whitney U | Significant (p < 0.001) — Action median $88M vs Drama $38M |
| 5 | Budget and revenue are correlated | Spearman Correlation | Significant (r = 0.69, p < 0.001) |

## How to Reproduce
```bash
pip install -r requirements.txt
python analysis.py
```

## Tools & Technologies
- Python 3
- pandas, numpy, matplotlib, seaborn, scipy
