# DSA 210 - Analyzing Movie Success: What Makes a Movie Profitable?

## Motivation
The film industry generates billions annually, yet predicting commercial success remains a challenge. This project uses data-driven methods to identify which measurable factors — budget, genre, cast, timing, ratings — most influence a movie's profitability.

## Data Sources
- **TMDB 5000 Movies Dataset** (Primary): ~5,000 movies with budget, revenue, genres, release date, ratings, popularity, and runtime. [Kaggle Link](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **TMDB Credits Dataset** (Enrichment): Cast and crew details — director, lead actor, cast size — merged with the primary dataset.

## Project Structure
```
├── movies_cleaned_light.csv      # Cleaned dataset (3,196 movies, 21 features)
├── plots/                        # All visualizations (18 plots)
├── analysis.py                   # EDA & hypothesis testing
├── ml_analysis.py                # Machine learning models
├── final_report.docx             # Final project report
├── requirements.txt              # Python dependencies
└── README.md
```

## Key Findings

### 1. Budget is necessary but not sufficient
Budget is the strongest predictor of gross revenue (r = 0.69), but low-budget movies actually have **higher ROI** than high-budget ones. Spending more increases absolute returns but decreases efficiency.

### 2. Visibility matters more than quality
Vote count and popularity predict revenue far better than vote average. A mediocre movie seen by millions will out-earn an excellent movie seen by thousands.

### 3. Genre has asymmetric risk-return profiles
Horror and Animation deliver the best ROI; Action and Adventure generate the highest gross revenue. Studios face a genuine efficiency-vs-scale tradeoff.

### 4. Timing matters
Summer releases (June–August) are statistically more profitable (p < 0.001), confirming the "summer blockbuster" effect.

### 5. Simple models match complex ones
Logistic Regression (F1 = 0.878, AUC = 0.814) outperformed Random Forest and Gradient Boosting for profitability classification, suggesting a largely linear decision boundary.

### 6. The industry has distinct segments
K-Means clustering revealed 4 movie archetypes: Low-Profile Films (53%), Critically Acclaimed (37%), Blockbusters (10%), and Mega Hits (<1%).

## Machine Learning Results

### Regression (Revenue Prediction)
| Model | R² | RMSE | CV R² |
|-------|-----|------|-------|
| Linear Regression | 0.440 | 1.495 | 0.314 |
| **Random Forest** | **0.673** | **1.144** | **0.608** |
| Gradient Boosting | 0.651 | 1.180 | 0.599 |

### Classification (Profitability Prediction)
| Model | Accuracy | F1 | AUC |
|-------|----------|-----|-----|
| **Logistic Regression** | **0.805** | **0.878** | **0.814** |
| Decision Tree | 0.736 | 0.827 | 0.691 |
| Random Forest | 0.794 | 0.869 | 0.803 |
| Gradient Boosting | 0.784 | 0.862 | 0.797 |

## Limitations
- ~35% of movies excluded due to missing budget/revenue data (biased toward mainstream films)
- Revenue is box-office only (excludes streaming, merchandise, home video)
- Financial figures not inflation-adjusted
- Post-release features (vote count, popularity) create temporal data leakage for prediction

## How to Reproduce
```bash
pip install -r requirements.txt
python analysis.py         # EDA & hypothesis tests
python ml_analysis.py      # ML models
```

## Tools
Python 3, pandas, numpy, matplotlib, seaborn, scipy, scikit-learn

## AI Disclosure
AI assistance (Claude by Anthropic) was used for code generation and report writing. All outputs were reviewed and validated by the student.
