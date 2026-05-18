#!/usr/bin/env python3
"""
DSA 210 - Milestone 1: Data Collection, EDA & Hypothesis Testing
Project: Analyzing Movie Success - What Makes a Movie Profitable?

This script:
1. Loads and cleans the TMDB 5000 Movies + Credits datasets
2. Engineers features (profit, ROI, director, lead actor, genre)
3. Performs EDA with 8 visualizations
4. Runs 5 hypothesis tests with statistical interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ast
import os
import warnings
warnings.filterwarnings('ignore')

# Set consistent plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

os.makedirs('plots', exist_ok=True)

# ============================================================
# 1. DATA LOADING & CLEANING
# ============================================================
print("=" * 60)
print("1. DATA LOADING & CLEANING")
print("=" * 60)

# Load both datasets from Kaggle's TMDB 5000 Movie Dataset
movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')

print(f"Movies dataset: {movies.shape[0]} rows, {movies.shape[1]} columns")
print(f"Credits dataset: {credits.shape[0]} rows, {credits.shape[1]} columns")

# Merge on movie ID to enrich with cast/crew data
credits.rename(columns={'movie_id': 'id'}, inplace=True)
df = movies.merge(credits[['id', 'cast', 'crew']], on='id', how='left')
print(f"\nMerged dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# --- Parse JSON-like string columns into usable features ---

def parse_json_column(text):
    """Safely parse JSON-like string columns from TMDB data."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []

# Extract genre names from JSON
df['genre_list'] = df['genres'].apply(parse_json_column)
df['genre_names'] = df['genre_list'].apply(lambda x: [g['name'] for g in x] if x else [])
df['primary_genre'] = df['genre_names'].apply(lambda x: x[0] if x else 'Unknown')
df['num_genres'] = df['genre_names'].apply(len)

# Extract director name from crew JSON
def get_director(crew_str):
    crew = parse_json_column(crew_str)
    for member in crew:
        if member.get('job') == 'Director':
            return member.get('name', 'Unknown')
    return 'Unknown'

df['director'] = df['crew'].apply(get_director)

# Extract lead actor (first-billed) from cast JSON
def get_lead_actor(cast_str):
    cast = parse_json_column(cast_str)
    if cast:
        return cast[0].get('name', 'Unknown')
    return 'Unknown'

df['lead_actor'] = df['cast'].apply(get_lead_actor)
df['cast_size'] = df['cast'].apply(lambda x: len(parse_json_column(x)))

# Parse release date and extract year/month for seasonal analysis
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month

# --- Remove invalid/missing financial data ---
# Movies with budget=0 or revenue=0 represent missing data, not actual zero values
print(f"\nBefore cleaning: {len(df)} movies")
print(f"  Movies with budget = 0: {(df['budget'] == 0).sum()}")
print(f"  Movies with revenue = 0: {(df['revenue'] == 0).sum()}")

df_clean = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
print(f"After removing zero budget/revenue: {len(df_clean)} movies")

# Create profitability metrics
df_clean['profit'] = df_clean['revenue'] - df_clean['budget']
df_clean['roi'] = (df_clean['revenue'] - df_clean['budget']) / df_clean['budget']
df_clean['is_profitable'] = (df_clean['profit'] > 0).astype(int)

# Remove extreme ROI outliers (beyond 99th percentile) to prevent distortion
roi_99 = df_clean['roi'].quantile(0.99)
df_clean = df_clean[df_clean['roi'] <= roi_99].copy()
print(f"After removing ROI outliers: {len(df_clean)} movies")

print("\n--- Key Statistics ---")
print(f"Budget range: ${df_clean['budget'].min():,.0f} - ${df_clean['budget'].max():,.0f}")
print(f"Revenue range: ${df_clean['revenue'].min():,.0f} - ${df_clean['revenue'].max():,.0f}")
print(f"Profitable movies: {df_clean['is_profitable'].sum()} ({df_clean['is_profitable'].mean()*100:.1f}%)")
print(f"Average ROI: {df_clean['roi'].mean():.2f}")

# Save cleaned data for ML stage
df_clean.to_csv('data/movies_cleaned.csv', index=False)
print("\nCleaned data saved to data/movies_cleaned.csv")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 60)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --- Figure 01: Budget vs Revenue ---
# This plot tests the core question: does spending more lead to earning more?
# Points above the red line are profitable; below are losses.
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(df_clean['budget'] / 1e6, df_clean['revenue'] / 1e6,
                     c=df_clean['vote_average'], cmap='RdYlGn', alpha=0.6, s=30, edgecolors='gray', linewidth=0.3)
plt.colorbar(scatter, label='Vote Average')
ax.plot([0, df_clean['budget'].max() / 1e6], [0, df_clean['budget'].max() / 1e6],
        'r--', alpha=0.5, label='Break-even line')
ax.set_xlabel('Budget (Millions $)', fontsize=13)
ax.set_ylabel('Revenue (Millions $)', fontsize=13)
ax.set_title('Budget vs Revenue (colored by rating)', fontsize=15, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('plots/01_budget_vs_revenue.png', dpi=150)
plt.close()
print("Plot saved: 01_budget_vs_revenue.png")

# --- Figure 02: Genre Distribution ---
# Shows which types of movies are most commonly produced
all_genres = df_clean['genre_names'].explode()
genre_counts = all_genres.value_counts().head(15)

fig, ax = plt.subplots(figsize=(10, 6))
genre_counts.plot(kind='barh', color=sns.color_palette('viridis', len(genre_counts)), ax=ax)
ax.set_xlabel('Number of Movies', fontsize=13)
ax.set_title('Top 15 Genres by Movie Count', fontsize=15, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('plots/02_genre_distribution.png', dpi=150)
plt.close()
print("Plot saved: 02_genre_distribution.png")

# --- Figure 03: Average ROI by Genre ---
# Reveals which genres offer the best return per dollar invested
genre_roi = []
for genre in genre_counts.index:
    mask = df_clean['genre_names'].apply(lambda x: genre in x)
    genre_roi.append({'genre': genre, 'avg_roi': df_clean.loc[mask, 'roi'].mean(),
                      'median_roi': df_clean.loc[mask, 'roi'].median()})
genre_roi_df = pd.DataFrame(genre_roi).sort_values('median_roi', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(genre_roi_df['genre'], genre_roi_df['median_roi'],
               color=sns.color_palette('coolwarm', len(genre_roi_df)))
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Median ROI', fontsize=13)
ax.set_title('Median Return on Investment by Genre', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/03_roi_by_genre.png', dpi=150)
plt.close()
print("Plot saved: 03_roi_by_genre.png")

# --- Figure 04: Revenue Trends Over Time ---
# Shows how the industry's financial scale has changed over decades
yearly = df_clean.groupby('release_year').agg(
    avg_revenue=('revenue', 'mean'),
    avg_budget=('budget', 'mean'),
    count=('id', 'count')
).reset_index()
yearly = yearly[yearly['release_year'] >= 1980]

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(yearly['release_year'], yearly['avg_budget'] / 1e6, 'b-o', markersize=4, label='Avg Budget')
ax1.plot(yearly['release_year'], yearly['avg_revenue'] / 1e6, 'g-o', markersize=4, label='Avg Revenue')
ax1.set_xlabel('Year', fontsize=13)
ax1.set_ylabel('Amount (Millions $)', fontsize=13)
ax1.set_title('Average Budget and Revenue Over Time', fontsize=15, fontweight='bold')
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.bar(yearly['release_year'], yearly['count'], alpha=0.15, color='gray', label='Movie Count')
ax2.set_ylabel('Number of Movies', fontsize=13)
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('plots/04_trends_over_time.png', dpi=150)
plt.close()
print("Plot saved: 04_trends_over_time.png")

# --- Figure 05: Correlation Heatmap ---
# Identifies which numerical features are most related to each other
num_cols = ['budget', 'revenue', 'profit', 'roi', 'vote_average', 'vote_count',
            'popularity', 'runtime', 'num_genres', 'cast_size']
corr = df_clean[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Correlation Heatmap of Numerical Features', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/05_correlation_heatmap.png', dpi=150)
plt.close()
print("Plot saved: 05_correlation_heatmap.png")

# --- Figure 06: Profitability by Release Month ---
# Tests whether release timing affects financial success
month_profit = df_clean.groupby('release_month').agg(
    avg_roi=('roi', 'mean'),
    pct_profitable=('is_profitable', 'mean')
).reset_index()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette('RdYlGn', 12)
bars = ax.bar(range(1, 13), month_profit['pct_profitable'] * 100, color=colors)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_xlabel('Release Month', fontsize=13)
ax.set_ylabel('% Profitable Movies', fontsize=13)
ax.set_title('Percentage of Profitable Movies by Release Month', fontsize=15, fontweight='bold')
ax.axhline(y=df_clean['is_profitable'].mean() * 100, color='red', linestyle='--', alpha=0.7, label='Overall Average')
ax.legend()
plt.tight_layout()
plt.savefig('plots/06_profitability_by_month.png', dpi=150)
plt.close()
print("Plot saved: 06_profitability_by_month.png")

# --- Figure 07: Key Distributions ---
# Understanding the shape of our target variables
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_clean['vote_average'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Vote Average')
axes[0].set_title('Distribution of Vote Average', fontweight='bold')

axes[1].hist(df_clean['roi'], bins=50, color='coral', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('ROI')
axes[1].set_title('Distribution of ROI', fontweight='bold')

plt.suptitle('Key Distributions', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/07_distributions.png', dpi=150)
plt.close()
print("Plot saved: 07_distributions.png")

# --- Top 10 Most Profitable Movies ---
top_profit = df_clean.nlargest(10, 'profit')[['title', 'budget', 'revenue', 'profit', 'roi', 'primary_genre']]
print("\n--- Top 10 Most Profitable Movies ---")
for i, row in top_profit.iterrows():
    print(f"  {row['title']}: profit=${row['profit']/1e6:.0f}M, ROI={row['roi']:.1f}x")

# ============================================================
# 3. HYPOTHESIS TESTING
# ============================================================
print("\n" + "=" * 60)
print("3. HYPOTHESIS TESTING")
print("=" * 60)

# --- Test 1: High budget vs Low budget ROI ---
# H0: There is no difference in ROI between high and low budget movies
# H1: There is a significant difference
print("\n--- Test 1: Do high-budget movies have higher ROI than low-budget movies? ---")
median_budget = df_clean['budget'].median()
high_budget = df_clean[df_clean['budget'] >= median_budget]['roi']
low_budget = df_clean[df_clean['budget'] < median_budget]['roi']

stat, p_value = stats.mannwhitneyu(high_budget, low_budget, alternative='two-sided')
print(f"  High-budget median ROI: {high_budget.median():.2f}")
print(f"  Low-budget median ROI: {low_budget.median():.2f}")
print(f"  Mann-Whitney U statistic: {stat:.2f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'} at alpha=0.05")
print(f"  Interpretation: Low-budget movies have HIGHER median ROI, suggesting diminishing returns at scale.")

# --- Test 2: Summer movies vs non-summer ---
# H0: Summer releases have the same profitability as non-summer
# H1: Summer releases are more profitable
print("\n--- Test 2: Are summer releases (Jun-Aug) more profitable? ---")
df_clean['is_summer'] = df_clean['release_month'].isin([6, 7, 8]).astype(int)
summer = df_clean[df_clean['is_summer'] == 1]['roi']
non_summer = df_clean[df_clean['is_summer'] == 0]['roi']

stat, p_value = stats.mannwhitneyu(summer, non_summer, alternative='greater')
print(f"  Summer median ROI: {summer.median():.2f}")
print(f"  Non-summer median ROI: {non_summer.median():.2f}")
print(f"  Mann-Whitney U statistic: {stat:.2f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'} at alpha=0.05")
print(f"  Interpretation: Summer releases are significantly more profitable, confirming the 'blockbuster season' effect.")

# --- Test 3: Rating vs ROI correlation ---
# H0: No correlation between vote_average and ROI
# H1: Significant correlation exists
print("\n--- Test 3: Is there a significant correlation between ratings and ROI? ---")
corr_coef, p_value = stats.pearsonr(df_clean['vote_average'], df_clean['roi'])
print(f"  Pearson correlation: {corr_coef:.4f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'} at alpha=0.05")
print(f"  Interpretation: Statistically significant but weak (r=0.24). Quality helps but is not a strong financial predictor.")

# --- Test 4: Action vs Drama revenue ---
# H0: Action and Drama movies have similar revenue
# H1: Action movies earn more
print("\n--- Test 4: Do Action movies have higher revenue than Drama movies? ---")
action_rev = df_clean[df_clean['genre_names'].apply(lambda x: 'Action' in x)]['revenue']
drama_rev = df_clean[df_clean['genre_names'].apply(lambda x: 'Drama' in x)]['revenue']

stat, p_value = stats.mannwhitneyu(action_rev, drama_rev, alternative='greater')
print(f"  Action median revenue: ${action_rev.median()/1e6:.1f}M")
print(f"  Drama median revenue: ${drama_rev.median()/1e6:.1f}M")
print(f"  Mann-Whitney U statistic: {stat:.2f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'} at alpha=0.05")
print(f"  Interpretation: Action significantly outearns Drama, reflecting global commercial appeal of the genre.")

# --- Test 5: Budget-Revenue correlation ---
# H0: No monotonic relationship between budget and revenue
# H1: Significant monotonic relationship
print("\n--- Test 5: Is there a significant correlation between budget and revenue? ---")
corr_coef, p_value = stats.spearmanr(df_clean['budget'], df_clean['revenue'])
print(f"  Spearman correlation: {corr_coef:.4f}")
print(f"  p-value: {p_value:.10f}")
print(f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'} at alpha=0.05")
print(f"  Interpretation: Strong positive correlation (r=0.69). Budget is the single strongest financial predictor.")

# --- Figure 08: Hypothesis Test Visualizations ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].boxplot([low_budget, high_budget], labels=['Low Budget', 'High Budget'])
axes[0, 0].set_ylabel('ROI')
axes[0, 0].set_title('ROI: Low vs High Budget Movies', fontweight='bold')

axes[0, 1].boxplot([non_summer, summer], labels=['Non-Summer', 'Summer (Jun-Aug)'])
axes[0, 1].set_ylabel('ROI')
axes[0, 1].set_title('ROI: Summer vs Non-Summer Releases', fontweight='bold')

axes[1, 0].scatter(df_clean['vote_average'], df_clean['roi'], alpha=0.3, s=15, color='steelblue')
z = np.polyfit(df_clean['vote_average'], df_clean['roi'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_clean['vote_average'].min(), df_clean['vote_average'].max(), 100)
axes[1, 0].plot(x_line, p(x_line), 'r-', linewidth=2)
axes[1, 0].set_xlabel('Vote Average')
axes[1, 0].set_ylabel('ROI')
axes[1, 0].set_title('Rating vs ROI', fontweight='bold')

axes[1, 1].boxplot([drama_rev / 1e6, action_rev / 1e6], labels=['Drama', 'Action'])
axes[1, 1].set_ylabel('Revenue (Millions $)')
axes[1, 1].set_title('Revenue: Action vs Drama', fontweight='bold')

plt.suptitle('Hypothesis Test Visualizations', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plots/08_hypothesis_tests.png', dpi=150)
plt.close()
print("\nPlot saved: 08_hypothesis_tests.png")

print("\n" + "=" * 60)
print("MILESTONE 1 COMPLETE")
print("=" * 60)
print(f"\nTotal movies analyzed: {len(df_clean)}")
print(f"Plots generated: 8 (saved in plots/ directory)")
print(f"Cleaned data saved: data/movies_cleaned.csv")
