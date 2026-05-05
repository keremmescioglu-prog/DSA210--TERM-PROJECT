#!/usr/bin/env python3
"""
DSA 210 - Milestone 2: Machine Learning Methods
Project: Analyzing Movie Success - What Makes a Movie Profitable?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

os.makedirs('plots', exist_ok=True)

# ============================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================
print("=" * 60)
print("1. DATA LOADING & FEATURE ENGINEERING")
print("=" * 60)

df = pd.read_csv('data/movies_cleaned_light.csv')
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# --- Feature Engineering ---

# Encode primary genre
top_genres = df['primary_genre'].value_counts().head(10).index.tolist()
df['genre_encoded'] = df['primary_genre'].apply(lambda x: x if x in top_genres else 'Other')
genre_dummies = pd.get_dummies(df['genre_encoded'], prefix='genre', drop_first=True)
df = pd.concat([df, genre_dummies], axis=1)

# Encode original language (English vs non-English)
df['is_english'] = (df['original_language'] == 'en').astype(int)

# Log-transform budget and revenue (to handle skewness)
df['log_budget'] = np.log1p(df['budget'])
df['log_revenue'] = np.log1p(df['revenue'])

# Budget categories
df['budget_category'] = pd.qcut(df['budget'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

# Feature columns for ML
feature_cols = ['budget', 'vote_average', 'vote_count', 'popularity',
                'runtime', 'num_genres', 'cast_size', 'is_english',
                'release_month'] + [c for c in genre_dummies.columns]

# Remove rows with NaN in features
df_ml = df.dropna(subset=feature_cols + ['revenue', 'is_profitable'])
print(f"ML-ready samples: {len(df_ml)}")
print(f"Features: {len(feature_cols)}")

X = df_ml[feature_cols].values
y_revenue = df_ml['revenue'].values
y_log_revenue = df_ml['log_revenue'].values
y_profitable = df_ml['is_profitable'].values.astype(int)

# Train-test split (80/20)
X_train, X_test, y_rev_train, y_rev_test = train_test_split(
    X, y_log_revenue, test_size=0.2, random_state=42)

_, _, y_prof_train, y_prof_test = train_test_split(
    X, y_profitable, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Profitable ratio (train): {y_prof_train.mean():.2%}")
print(f"Profitable ratio (test): {y_prof_test.mean():.2%}")

# ============================================================
# 2. REGRESSION: Predicting Movie Revenue
# ============================================================
print("\n" + "=" * 60)
print("2. REGRESSION: Predicting Movie Revenue (log-transformed)")
print("=" * 60)

regression_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

reg_results = {}

for name, model in regression_models.items():
    # Use scaled data for Linear Regression, raw for tree-based
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_rev_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_rev_train, cv=5, scoring='r2')
    else:
        model.fit(X_train, y_rev_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_rev_train, cv=5, scoring='r2')

    rmse = np.sqrt(mean_squared_error(y_rev_test, y_pred))
    mae = mean_absolute_error(y_rev_test, y_pred)
    r2 = r2_score(y_rev_test, y_pred)

    reg_results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'CV R² (mean)': cv_scores.mean(), 'predictions': y_pred}

    print(f"\n  {name}:")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE:  {mae:.4f}")
    print(f"    R²:   {r2:.4f}")
    print(f"    5-Fold CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# --- Regression: Actual vs Predicted Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, result) in enumerate(reg_results.items()):
    axes[i].scatter(y_rev_test, result['predictions'], alpha=0.4, s=20, color='steelblue')
    min_val = min(y_rev_test.min(), result['predictions'].min())
    max_val = max(y_rev_test.max(), result['predictions'].max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    axes[i].set_xlabel('Actual Log Revenue')
    axes[i].set_ylabel('Predicted Log Revenue')
    axes[i].set_title(f"{name}\nR² = {result['R²']:.3f}", fontweight='bold')
    axes[i].legend()
plt.suptitle('Regression: Actual vs Predicted Revenue', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/09_regression_predictions.png', dpi=150)
plt.close()
print("\nPlot saved: 09_regression_predictions.png")

# --- Feature Importance (from best tree model) ---
best_reg = regression_models['Gradient Boosting']
importances = best_reg.feature_importances_
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(feat_imp['feature'], feat_imp['importance'], color=sns.color_palette('viridis', len(feat_imp)))
ax.set_xlabel('Feature Importance', fontsize=13)
ax.set_title('Top 15 Features for Revenue Prediction (Gradient Boosting)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/10_feature_importance_regression.png', dpi=150)
plt.close()
print("Plot saved: 10_feature_importance_regression.png")

# --- Regression Model Comparison ---
reg_comparison = pd.DataFrame({
    name: {k: v for k, v in result.items() if k != 'predictions'}
    for name, result in reg_results.items()
}).T

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['R²', 'RMSE', 'MAE']
colors = ['#2ecc71', '#e74c3c', '#3498db']
for i, metric in enumerate(metrics):
    bars = axes[i].bar(reg_comparison.index, reg_comparison[metric], color=colors[i], alpha=0.8)
    axes[i].set_title(metric, fontweight='bold', fontsize=14)
    axes[i].tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, reg_comparison[metric]):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', fontsize=11)
plt.suptitle('Regression Model Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/11_regression_comparison.png', dpi=150)
plt.close()
print("Plot saved: 11_regression_comparison.png")

# ============================================================
# 3. CLASSIFICATION: Predicting Profitability
# ============================================================
print("\n" + "=" * 60)
print("3. CLASSIFICATION: Predicting Movie Profitability")
print("=" * 60)

classification_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
}

clf_results = {}

for name, model in classification_models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_prof_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        cv_scores = cross_val_score(model, X_train_scaled, y_prof_train, cv=5, scoring='accuracy')
    else:
        model.fit(X_train, y_prof_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(model, X_train, y_prof_train, cv=5, scoring='accuracy')

    acc = accuracy_score(y_prof_test, y_pred)
    prec = precision_score(y_prof_test, y_pred)
    rec = recall_score(y_prof_test, y_pred)
    f1 = f1_score(y_prof_test, y_pred)
    fpr, tpr, _ = roc_curve(y_prof_test, y_prob)
    roc_auc = auc(fpr, tpr)

    clf_results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1': f1, 'AUC': roc_auc, 'CV Accuracy': cv_scores.mean(),
        'predictions': y_pred, 'probabilities': y_prob,
        'fpr': fpr, 'tpr': tpr
    }

    print(f"\n  {name}:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"    AUC:       {roc_auc:.4f}")
    print(f"    5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# --- ROC Curves ---
fig, ax = plt.subplots(figsize=(8, 7))
colors_roc = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
for i, (name, result) in enumerate(clf_results.items()):
    ax.plot(result['fpr'], result['tpr'], color=colors_roc[i], linewidth=2,
            label=f"{name} (AUC = {result['AUC']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.500)')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves - Profitability Classification', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig('plots/12_roc_curves.png', dpi=150)
plt.close()
print("\nPlot saved: 12_roc_curves.png")

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for i, (name, result) in enumerate(clf_results.items()):
    cm = confusion_matrix(y_prof_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Not Profitable', 'Profitable'],
                yticklabels=['Not Profitable', 'Profitable'])
    axes[i].set_title(f'{name}\nAcc: {result["Accuracy"]:.3f}', fontweight='bold', fontsize=11)
    axes[i].set_ylabel('Actual')
    axes[i].set_xlabel('Predicted')
plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/13_confusion_matrices.png', dpi=150)
plt.close()
print("Plot saved: 13_confusion_matrices.png")

# --- Classification Model Comparison ---
clf_comparison = pd.DataFrame({
    name: {k: v for k, v in result.items() if k not in ['predictions', 'probabilities', 'fpr', 'tpr']}
    for name, result in clf_results.items()
}).T

fig, ax = plt.subplots(figsize=(12, 6))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
x = np.arange(len(clf_comparison))
width = 0.15
colors_bar = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
for i, metric in enumerate(metrics_to_plot):
    ax.bar(x + i * width, clf_comparison[metric], width, label=metric, color=colors_bar[i], alpha=0.85)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(clf_comparison.index, fontsize=11)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Classification Model Comparison', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0.5, 1.05)
plt.tight_layout()
plt.savefig('plots/14_classification_comparison.png', dpi=150)
plt.close()
print("Plot saved: 14_classification_comparison.png")

# --- Feature Importance (Classification) ---
best_clf = classification_models['Gradient Boosting']
importances_clf = best_clf.feature_importances_
feat_imp_clf = pd.DataFrame({'feature': feature_cols, 'importance': importances_clf})
feat_imp_clf = feat_imp_clf.sort_values('importance', ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(feat_imp_clf['feature'], feat_imp_clf['importance'], color=sns.color_palette('magma', len(feat_imp_clf)))
ax.set_xlabel('Feature Importance', fontsize=13)
ax.set_title('Top 15 Features for Profitability Prediction (Gradient Boosting)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/15_feature_importance_classification.png', dpi=150)
plt.close()
print("Plot saved: 15_feature_importance_classification.png")

# ============================================================
# 4. CLUSTERING: Discovering Movie Groups
# ============================================================
print("\n" + "=" * 60)
print("4. CLUSTERING: Discovering Movie Groups")
print("=" * 60)

cluster_features = ['budget', 'revenue', 'vote_average', 'popularity', 'runtime']
X_cluster = df_ml[cluster_features].dropna().values
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# Elbow method
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Clusters (K)', fontsize=13)
ax.set_ylabel('Inertia', fontsize=13)
ax.set_title('Elbow Method for Optimal K', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/16_elbow_method.png', dpi=150)
plt.close()
print("Plot saved: 16_elbow_method.png")

# Apply K-Means with K=4
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_cluster_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_scaled)

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='Set1', alpha=0.6, s=30)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=13)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=13)
ax.set_title(f'Movie Clusters (K={optimal_k}) - PCA Visualization', fontsize=15, fontweight='bold')
plt.colorbar(scatter, label='Cluster', ticks=range(optimal_k))
plt.tight_layout()
plt.savefig('plots/17_clusters_pca.png', dpi=150)
plt.close()
print("Plot saved: 17_clusters_pca.png")

# Cluster profiles
df_clustered = df_ml[cluster_features].dropna().copy()
df_clustered['cluster'] = clusters

print("\n--- Cluster Profiles ---")
cluster_profiles = df_clustered.groupby('cluster').agg({
    'budget': 'median',
    'revenue': 'median',
    'vote_average': 'mean',
    'popularity': 'mean',
    'runtime': 'mean'
}).round(2)

cluster_sizes = df_clustered['cluster'].value_counts().sort_index()
cluster_profiles['count'] = cluster_sizes.values

labels = []
for idx, row in cluster_profiles.iterrows():
    if row['budget'] > cluster_profiles['budget'].median() and row['revenue'] > cluster_profiles['revenue'].median():
        label = 'Big Budget Blockbusters'
    elif row['vote_average'] > cluster_profiles['vote_average'].median() and row['budget'] < cluster_profiles['budget'].median():
        label = 'Critically Acclaimed Indies'
    elif row['popularity'] > cluster_profiles['popularity'].median():
        label = 'Popular Mainstream'
    else:
        label = 'Low-Profile Films'
    labels.append(label)
    print(f"\n  Cluster {idx} - {label} ({row['count']:.0f} movies):")
    print(f"    Median Budget:  ${row['budget']/1e6:.1f}M")
    print(f"    Median Revenue: ${row['revenue']/1e6:.1f}M")
    print(f"    Avg Rating:     {row['vote_average']:.1f}")
    print(f"    Avg Popularity: {row['popularity']:.1f}")
    print(f"    Avg Runtime:    {row['runtime']:.0f} min")

# Cluster profile heatmap
fig, ax = plt.subplots(figsize=(10, 5))
profile_normalized = cluster_profiles.drop('count', axis=1).copy()
for col in profile_normalized.columns:
    profile_normalized[col] = (profile_normalized[col] - profile_normalized[col].min()) / \
                               (profile_normalized[col].max() - profile_normalized[col].min())
profile_normalized.index = [f"Cluster {i}\n{labels[i]}" for i in range(optimal_k)]
sns.heatmap(profile_normalized, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, linewidths=1)
ax.set_title('Cluster Profiles (Normalized)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/18_cluster_profiles.png', dpi=150)
plt.close()
print("\nPlot saved: 18_cluster_profiles.png")

# ============================================================
# 5. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("5. MILESTONE 2 SUMMARY")
print("=" * 60)

print("\n--- Regression (Revenue Prediction) ---")
print(f"  Best model: Gradient Boosting (R² = {reg_results['Gradient Boosting']['R²']:.3f})")
print(f"  Top predictors: budget, popularity, vote_count")

print("\n--- Classification (Profitability) ---")
best_clf_name = max(clf_results, key=lambda x: clf_results[x]['F1'])
print(f"  Best model: {best_clf_name} (F1 = {clf_results[best_clf_name]['F1']:.3f}, AUC = {clf_results[best_clf_name]['AUC']:.3f})")
print(f"  Top predictors: budget, popularity, vote_count")

print("\n--- Clustering ---")
print(f"  Optimal K: {optimal_k}")
print(f"  Groups: {', '.join(labels)}")

print(f"\nNew plots generated: 10 (plots 09-18)")
print("MILESTONE 2 COMPLETE")
