# DSA 210 - Analyzing Movie Success: What Makes a Movie Profitable?

## Project Description
This project investigates which factors — such as budget, genre, cast, release timing, and audience ratings — are most strongly associated with a movie's commercial success. Using the TMDB 5000 Movies dataset enriched with cast/crew credits data, I analyze patterns in profitability and apply machine learning to predict movie success.

## Data Sources
- **TMDB 5000 Movies Dataset** (Primary): ~5,000 movies with budget, revenue, genres, release date, ratings, popularity, and runtime. [Kaggle Link](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **TMDB Credits Dataset** (Enrichment): Cast and crew details for each movie, enabling analysis of director/actor effects on profitability.

## Project Structure
```
├── movies_cleaned_light.csv      # Cleaned & feature-engineered dataset
├── plots/                        # All generated visualizations
├── analysis.py                   # Milestone 1: EDA & hypothesis testing
├── ml_analysis.py                # Milestone 2: Machine learning models
├── requirements.txt              # Python dependencies
└── README.md
```

## Milestone 1: EDA & Hypothesis Testing

### EDA Highlights
- **3,196 movies** analyzed after cleaning (removed zero budget/revenue entries)
- **75.3%** of movies are profitable, with an average ROI of 3.02x
- Budget and revenue have a strong positive correlation (Spearman r = 0.69)
- Drama and Comedy are the most common genres; Horror and Animation show highest median ROI

### Hypothesis Tests
| # | Hypothesis | Test | Result |
|---|-----------|------|--------|
| 1 | High-budget movies have higher ROI | Mann-Whitney U | Low-budget movies actually have *higher* median ROI |
| 2 | Summer releases are more profitable | Mann-Whitney U | Significant (p < 0.001) |
| 3 | Higher ratings correlate with higher ROI | Pearson Correlation | Significant (r = 0.24, p < 0.001) |
| 4 | Action movies earn more revenue than Drama | Mann-Whitney U | Significant — Action $88M vs Drama $38M |
| 5 | Budget and revenue are correlated | Spearman Correlation | Significant (r = 0.69, p < 0.001) |

## Milestone 2: Machine Learning

### Regression — Predicting Revenue
| Model | R² | RMSE | MAE | CV R² |
|-------|-----|------|-----|-------|
| Linear Regression | 0.440 | 1.495 | 1.043 | 0.314 |
| Random Forest | 0.673 | 1.144 | 0.791 | 0.608 |
| Gradient Boosting | 0.651 | 1.180 | 0.805 | 0.599 |

**Best model:** Random Forest (R² = 0.673)

### Classification — Predicting Profitability
| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Logistic Regression | 0.805 | 0.827 | 0.935 | 0.878 | 0.814 |
| Decision Tree | 0.736 | 0.812 | 0.844 | 0.827 | 0.691 |
| Random Forest | 0.794 | 0.828 | 0.915 | 0.869 | 0.803 |
| Gradient Boosting | 0.784 | 0.829 | 0.898 | 0.862 | 0.797 |

**Best model:** Logistic Regression (F1 = 0.878, AUC = 0.814)

### Clustering — Movie Groups
Using K-Means (K=4), movies were grouped into:
1. **Low-Profile Films** (1,684 movies) — Low budget, low revenue, average ratings
2. **Big Budget Blockbusters** (319 movies) — High budget ($139M), high revenue ($459M)
3. **Critically Acclaimed Indies** (1,187 movies) — Lower budget, higher ratings (7.0)
4. **Mega Hits** (6 movies) — Extreme popularity and revenue ($778M)

### Key Findings
- **Budget** and **popularity** are the strongest predictors of both revenue and profitability
- Logistic Regression outperforms complex models for profitability classification
- Random Forest provides the best revenue predictions (R² = 0.673)
- Movies naturally cluster into distinct groups based on budget-revenue-rating profiles

## How to Reproduce
```bash
pip install -r requirements.txt
python analysis.py         # Milestone 1: EDA & hypothesis tests
python ml_analysis.py      # Milestone 2: ML models
```

## Tools & Technologies
- Python 3
- pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
