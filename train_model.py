"""
IPL Match Winner Predictor - Machine Learning Training & Evaluation Pipeline
=============================================================================
This script implements a rigorous, leak-free machine learning pipeline for
predicting the winner of Indian Premier League (IPL) matches.

Key Features:
- Leakage-Free Pre-Match Feature Engineering (Strictly Prior Chronological Stats)
- Time-Based Split (Train on Earlier Seasons, Test on Latest Seasons)
- Stratified Random Split (Secondary Comparison)
- Domain Baselines (Toss Winner, Better Historical Win Rate)
- Model Comparison: Logistic Regression, Random Forest, Gradient Boosting (Pipeline)
- 5-Fold TimeSeriesSplit Cross-Validation & Hyperparameter Tuning (GridSearchCV)
- Evaluation Visuals: Confusion Matrix, ROC Curve, Feature Importance
- Team Clustering: KMeans on Team Aggregates (Win Rate, Avg Runs, Avg Wickets)
"""

import os
import json
import warnings
from collections import defaultdict
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, silhouette_score
)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split, cross_val_score
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Standardize franchise name changes for continuous historical tracking
TEAM_NAME_MAPPING = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
}

def map_season(match_id):
    """Map match_id to IPL season based on Cricsheet / Cricinfo match numbering."""
    if match_id <= 336040: return 2008
    elif match_id <= 392239: return 2009
    elif match_id <= 419165: return 2010
    elif match_id <= 501271: return 2011
    elif match_id <= 548381: return 2012
    elif match_id <= 601676: return 2013
    elif match_id <= 734049: return 2014
    elif match_id <= 829823: return 2015
    elif match_id <= 981019: return 2016
    elif match_id <= 1082650: return 2017
    elif match_id <= 1136620: return 2018
    elif match_id <= 1181768: return 2019
    elif match_id <= 1237181: return 2020
    elif match_id <= 1254117: return 2021
    elif match_id <= 1312200: return 2022
    elif match_id <= 1370353: return 2023
    elif match_id <= 1426312: return 2024
    elif match_id <= 1485779: return 2025
    else: return 2026

def load_and_preprocess_data():
    """
    Load data from raw ball-by-ball IPL.csv if available; otherwise load the
    bundled match_history.pkl artifact. Decode and clean data.
    """
    print("=" * 70)
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("=" * 70)
    
    if os.path.exists('IPL.csv'):
        print("Found 'IPL.csv'. Aggregating ball-by-ball records...")
        raw_df = pd.read_csv('IPL.csv', low_memory=False)
        match_df = raw_df.groupby('match_id').agg(
            total_runs=('runs_total', 'sum'),
            total_wickets=('player_out', lambda x: x.notna().sum()),
            Venue=('venue', 'first'),
            Toss_Winner=('toss_winner', 'first'),
            Toss_Decision=('toss_decision', 'first'),
            Team_A=('batting_team', 'first'),
            Team_B=('bowling_team', 'first'),
            Winner=('match_won_by', 'first')
        ).reset_index()
    elif os.path.exists('match_history.pkl'):
        print("Loading match records from 'match_history.pkl'...")
        match_df = pd.read_pickle('match_history.pkl')
        # If integer encoded, decode using encoders.pkl
        if os.path.exists('encoders.pkl') and match_df['Winner'].dtype != object:
            encoders = joblib.load('encoders.pkl')
            for col in ['Venue', 'Toss_Winner', 'Toss_Decision', 'Team_A', 'Team_B', 'Winner']:
                if col in match_df.columns and col in encoders:
                    match_df[col] = encoders[col].inverse_transform(match_df[col])
    else:
        raise FileNotFoundError("Neither IPL.csv nor match_history.pkl found in workspace!")

    # Drop unplayable or abandoned matches (Winner == 'Unknown' or NaN)
    initial_count = len(match_df)
    match_df = match_df.dropna(subset=['Winner', 'Venue', 'Toss_Winner', 'Toss_Decision', 'Team_A', 'Team_B'])
    match_df = match_df[match_df['Winner'] != 'Unknown'].copy().reset_index(drop=True)
    
    # Standardize renames
    for col in ['Team_A', 'Team_B', 'Toss_Winner', 'Winner']:
        match_df[col] = match_df[col].replace(TEAM_NAME_MAPPING)
        
    match_df['Season'] = match_df['match_id'].apply(map_season)
    
    # Sort chronologically by match_id
    match_df = match_df.sort_values(by='match_id').reset_index(drop=True)
    print(f"Total raw matches: {initial_count} | Decided matches retained: {len(match_df)}")
    print(f"Seasons covered: {match_df['Season'].min()} to {match_df['Season'].max()}")
    return match_df

def build_leakage_free_features(match_df):
    """
    Construct pre-match features using STRICTLY historical data prior to each match.
    Zero leakage: no final runs, no final wickets, no post-match statistics.
    Also builds symmetric perspectives for balanced, order-invariant training.
    """
    print("\n" + "=" * 70)
    print("STEP 2: LEAKAGE-FREE PRE-MATCH FEATURE ENGINEERING")
    print("=" * 70)
    
    team_matches = defaultdict(list)
    team_venue_matches = defaultdict(list)
    h2h_matches = defaultdict(list)
    venue_bat_first_wins = defaultdict(list)
    venue_toss_wins = defaultdict(list)

    symmetric_rows = []
    single_rows = []

    for idx, row in match_df.iterrows():
        tA = row['Team_A'] # Batting team in inning 1
        tB = row['Team_B'] # Bowling team in inning 1
        ven = row['Venue']
        tw = row['Toss_Winner']
        td = row['Toss_Decision']
        winner = row['Winner']
        season = row['Season']
        mid = row['match_id']

        # 1. Historical Win Rates strictly prior to match idx
        tA_hist = team_matches[tA]
        tB_hist = team_matches[tB]
        tA_wr = np.mean(tA_hist) if len(tA_hist) > 0 else 0.50
        tB_wr = np.mean(tB_hist) if len(tB_hist) > 0 else 0.50

        # 2. Recent Form (last 5 matches strictly prior)
        tA_form = np.mean(tA_hist[-5:]) if len(tA_hist) > 0 else 0.50
        tB_form = np.mean(tB_hist[-5:]) if len(tB_hist) > 0 else 0.50

        # 3. Head-to-Head strictly prior
        h2h_key = tuple(sorted([tA, tB]))
        h2h_prior = h2h_matches[h2h_key]
        if len(h2h_prior) > 0:
            h2h_a_wins = sum(1 for w in h2h_prior if w == tA)
            h2h_win_rate_a = h2h_a_wins / len(h2h_prior)
        else:
            h2h_win_rate_a = 0.50
        h2h_count = len(h2h_prior)

        # 4. Venue stats strictly prior
        tA_ven = team_venue_matches[(tA, ven)]
        tB_ven = team_venue_matches[(tB, ven)]
        tA_ven_rate = np.mean(tA_ven) if len(tA_ven) > 0 else tA_wr
        tB_ven_rate = np.mean(tB_ven) if len(tB_ven) > 0 else tB_wr

        v_bf = venue_bat_first_wins[ven]
        venue_bat_first_rate = np.mean(v_bf) if len(v_bf) > 0 else 0.45

        v_tw = venue_toss_wins[ven]
        venue_toss_win_rate = np.mean(v_tw) if len(v_tw) > 0 else 0.50

        # Pre-match toss & batting first indicators
        toss_win_A = 1 if tw == tA else 0
        toss_dec_field = 1 if td == 'field' else 0
        target_A = 1 if winner == tA else 0

        # Perspective 1: Team A vs Team B
        f1 = {
            'match_id': mid,
            'Season': season,
            'Team_A': tA,
            'Team_B': tB,
            'Venue': ven,
            'Toss_Winner': tw,
            'Toss_Decision': td,
            'team_a_batting_first': 1,
            'toss_winner_is_team_a': toss_win_A,
            'toss_decision_field': toss_dec_field,
            'team_a_win_rate': tA_wr,
            'team_b_win_rate': tB_wr,
            'diff_win_rate': tA_wr - tB_wr,
            'team_a_form': tA_form,
            'team_b_form': tB_form,
            'diff_form': tA_form - tB_form,
            'h2h_win_rate_a': h2h_win_rate_a,
            'h2h_count': h2h_count,
            'venue_win_rate_a': tA_ven_rate,
            'venue_win_rate_b': tB_ven_rate,
            'diff_venue_rate': tA_ven_rate - tB_ven_rate,
            'venue_bat_first_rate': venue_bat_first_rate,
            'venue_toss_win_rate': venue_toss_win_rate,
            'target': target_A
        }
        single_rows.append(f1)
        symmetric_rows.append(f1)

        # Perspective 2: Team B vs Team A (Reciprocal, Symmetrized)
        f2 = {
            'match_id': mid,
            'Season': season,
            'Team_A': tB,
            'Team_B': tA,
            'Venue': ven,
            'Toss_Winner': tw,
            'Toss_Decision': td,
            'team_a_batting_first': 0,
            'toss_winner_is_team_a': 1 - toss_win_A,
            'toss_decision_field': toss_dec_field,
            'team_a_win_rate': tB_wr,
            'team_b_win_rate': tA_wr,
            'diff_win_rate': tB_wr - tA_wr,
            'team_a_form': tB_form,
            'team_b_form': tA_form,
            'diff_form': tB_form - tA_form,
            'h2h_win_rate_a': 1.0 - h2h_win_rate_a,
            'h2h_count': h2h_count,
            'venue_win_rate_a': tB_ven_rate,
            'venue_win_rate_b': tA_ven_rate,
            'diff_venue_rate': tB_ven_rate - tA_ven_rate,
            'venue_bat_first_rate': venue_bat_first_rate,
            'venue_toss_win_rate': venue_toss_win_rate,
            'target': 1 - target_A
        }
        symmetric_rows.append(f2)

        # Update historical trackers STRICTLY AFTER computing pre-match features
        team_matches[tA].append(1 if winner == tA else 0)
        team_matches[tB].append(1 if winner == tB else 0)
        team_venue_matches[(tA, ven)].append(1 if winner == tA else 0)
        team_venue_matches[(tB, ven)].append(1 if winner == tB else 0)
        h2h_matches[h2h_key].append(winner)
        venue_bat_first_wins[ven].append(1 if winner == tA else 0)
        venue_toss_wins[ven].append(1 if winner == tw else 0)

    single_df = pd.DataFrame(single_rows)
    sym_df = pd.DataFrame(symmetric_rows)
    print(f"Extracted {len(single_df)} single matches and {len(sym_df)} symmetric contest rows.")
    print(f"Target class balance in symmetric dataset: {sym_df['target'].mean() * 100:.1f}%")
    return single_df, sym_df

def compute_baselines(match_df, test_season_threshold=2024):
    """
    Evaluate domain baseline models:
    (a) Always predict the toss winner
    (b) Always predict the team with better historical win rate
    """
    print("\n" + "=" * 70)
    print("STEP 3: DOMAIN BASELINE EVALUATION")
    print("=" * 70)
    
    team_wins = defaultdict(int)
    team_matches = defaultdict(int)

    actuals = []
    toss_preds = []
    bwr_preds = []
    seasons = []

    for idx, row in match_df.iterrows():
        tA = row['Team_A']
        tB = row['Team_B']
        tw = row['Toss_Winner']
        winner = row['Winner']
        season = row['Season']

        wr_A = team_wins[tA] / team_matches[tA] if team_matches[tA] > 0 else 0.50
        wr_B = team_wins[tB] / team_matches[tB] if team_matches[tB] > 0 else 0.50

        # Predict team with better historical win rate (toss winner as tiebreaker)
        if wr_A > wr_B:
            pred_bwr = tA
        elif wr_B > wr_A:
            pred_bwr = tB
        else:
            pred_bwr = tw

        actuals.append(winner)
        toss_preds.append(tw)
        bwr_preds.append(pred_bwr)
        seasons.append(season)

        # Update stats
        team_matches[tA] += 1
        team_matches[tB] += 1
        team_wins[winner] += 1

    b_df = pd.DataFrame({
        'Season': seasons,
        'Actual': actuals,
        'Pred_Toss': toss_preds,
        'Pred_BWR': bwr_preds
    })

    b_df['Toss_Correct'] = b_df['Pred_Toss'] == b_df['Actual']
    b_df['BWR_Correct'] = b_df['Pred_BWR'] == b_df['Actual']

    all_toss_acc = b_df['Toss_Correct'].mean()
    all_bwr_acc = b_df['BWR_Correct'].mean()

    test_mask = b_df['Season'] >= test_season_threshold
    test_toss_acc = b_df.loc[test_mask, 'Toss_Correct'].mean()
    test_bwr_acc = b_df.loc[test_mask, 'BWR_Correct'].mean()

    print(f"Overall Dataset (N={len(b_df)} matches):")
    print(f"  - Baseline (a) Always Toss Winner:          {all_toss_acc * 100:.2f}%")
    print(f"  - Baseline (b) Better Historical Win Rate:  {all_bwr_acc * 100:.2f}%")
    print(f"\nTest Set Seasons >= {test_season_threshold} (N={test_mask.sum()} matches):")
    print(f"  - Baseline (a) Always Toss Winner:          {test_toss_acc * 100:.2f}%")
    print(f"  - Baseline (b) Better Historical Win Rate:  {test_bwr_acc * 100:.2f}%")

    return {
        'all_toss_acc': all_toss_acc,
        'all_bwr_acc': all_bwr_acc,
        'test_toss_acc': test_toss_acc,
        'test_bwr_acc': test_bwr_acc,
        'test_count': int(test_mask.sum())
    }

def run_model_experiments(sym_df, test_season_threshold=2024):
    """
    Train and evaluate Logistic Regression, Random Forest, and Gradient Boosting
    in a Pipeline with proper ColumnTransformer encoding on:
    1. Time-based Split (Seasons < threshold vs >= threshold)
    2. Stratified Random Split (80/20)
    Saves results to results/metrics.csv.
    """
    print("\n" + "=" * 70)
    print("STEP 4: MODEL COMPARISON & PIPELINE TRAINING")
    print("=" * 70)
    
    num_cols = [
        'team_a_batting_first', 'toss_winner_is_team_a', 'toss_decision_field',
        'team_a_win_rate', 'team_b_win_rate', 'diff_win_rate',
        'team_a_form', 'team_b_form', 'diff_form',
        'h2h_win_rate_a', 'h2h_count',
        'venue_win_rate_a', 'venue_win_rate_b', 'diff_venue_rate',
        'venue_bat_first_rate', 'venue_toss_win_rate'
    ]
    cat_cols = ['Venue', 'Team_A', 'Team_B', 'Toss_Decision']
    feature_cols = num_cols + cat_cols

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=0.2, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=4, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42)
    }

    # 1. Time-based Split
    train_mask = sym_df['Season'] < test_season_threshold
    test_mask = sym_df['Season'] >= test_season_threshold

    X_train_time = sym_df.loc[train_mask, feature_cols]
    y_train_time = sym_df.loc[train_mask, 'target']
    X_test_time = sym_df.loc[test_mask, feature_cols]
    y_test_time = sym_df.loc[test_mask, 'target']

    # 2. Stratified Random Split (80/20)
    X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
        sym_df[feature_cols], sym_df['target'], test_size=0.20, random_state=42, stratify=sym_df['target']
    )

    results_records = []
    fitted_time_pipelines = {}
    time_predictions = {}

    print(f"\n--- TIME-BASED SPLIT (Train: Seasons < {test_season_threshold} [N={len(X_train_time)} rows], Test: Seasons >= {test_season_threshold} [N={len(X_test_time)} rows]) ---")
    for name, clf in models.items():
        pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
        pipe.fit(X_train_time, y_train_time)
        fitted_time_pipelines[name] = pipe

        y_pred = pipe.predict(X_test_time)
        y_prob = pipe.predict_proba(X_test_time)[:, 1]
        time_predictions[name] = {'y_pred': y_pred, 'y_prob': y_prob}

        acc = accuracy_score(y_test_time, y_pred)
        prec = precision_score(y_test_time, y_pred, zero_division=0)
        rec = recall_score(y_test_time, y_pred, zero_division=0)
        f1 = f1_score(y_test_time, y_pred, zero_division=0)
        auc = roc_auc_score(y_test_time, y_prob)

        print(f"  {name:20s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        results_records.append({
            'Split': f'Time-Based (>= {test_season_threshold})',
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'ROC-AUC': round(auc, 4)
        })

    print(f"\n--- STRATIFIED RANDOM SPLIT (80/20: Train N={len(X_train_strat)}, Test N={len(X_test_strat)}) ---")
    for name, clf in models.items():
        pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
        pipe.fit(X_train_strat, y_train_strat)

        y_pred = pipe.predict(X_test_strat)
        y_prob = pipe.predict_proba(X_test_strat)[:, 1]

        acc = accuracy_score(y_test_strat, y_pred)
        prec = precision_score(y_test_strat, y_pred, zero_division=0)
        rec = recall_score(y_test_strat, y_pred, zero_division=0)
        f1 = f1_score(y_test_strat, y_pred, zero_division=0)
        auc = roc_auc_score(y_test_strat, y_prob)

        print(f"  {name:20s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        results_records.append({
            'Split': 'Stratified Random (80/20)',
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'ROC-AUC': round(auc, 4)
        })

    metrics_df = pd.DataFrame(results_records)
    os.makedirs('results', exist_ok=True)
    metrics_df.to_csv('results/metrics.csv', index=False)
    print("\nSaved evaluation table to 'results/metrics.csv'.")

    return {
        'metrics_df': metrics_df,
        'pipelines': fitted_time_pipelines,
        'predictions': time_predictions,
        'X_train_time': X_train_time,
        'y_train_time': y_train_time,
        'X_test_time': X_test_time,
        'y_test_time': y_test_time,
        'feature_cols': feature_cols,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'preprocessor': preprocessor
    }

def run_cross_validation_and_tuning(exp_data):
    """
    Perform 5-fold TimeSeriesSplit cross validation on chronological training data.
    Tune hyperparameters of the best model using GridSearchCV.
    Report mean and standard deviation of CV scores before and after tuning.
    """
    print("\n" + "=" * 70)
    print("STEP 5: 5-FOLD TIME-SERIES CROSS VALIDATION & TUNING")
    print("=" * 70)
    
    X_train = exp_data['X_train_time']
    y_train = exp_data['y_train_time']
    preprocessor = exp_data['preprocessor']

    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_summary = {}
    print("Running 5-Fold TimeSeriesSplit Cross-Validation (Pre-Tuning)...")
    for name, pipe in exp_data['pipelines'].items():
        scores = cross_val_score(pipe, X_train, y_train, cv=tscv, scoring='accuracy')
        auc_scores = cross_val_score(pipe, X_train, y_train, cv=tscv, scoring='roc_auc')
        cv_summary[name] = {
            'mean_acc': float(np.mean(scores)),
            'std_acc': float(np.std(scores)),
            'mean_auc': float(np.mean(auc_scores)),
            'std_auc': float(np.std(auc_scores))
        }
        print(f"  {name:20s} | Mean CV Acc: {np.mean(scores):.4f} +/- {np.std(scores):.4f} | Mean AUC: {np.mean(auc_scores):.4f} +/- {np.std(auc_scores):.4f}")

    # Best model family for tuning: Random Forest
    print("\nRunning GridSearchCV on Random Forest...")
    rf_pipe = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
    param_grid = {
        'clf__n_estimators': [50, 100, 150],
        'clf__max_depth': [3, 5, 8],
        'clf__min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(rf_pipe, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_idx = grid_search.best_index_
    best_mean_acc = grid_search.cv_results_['mean_test_score'][best_idx]
    best_std_acc = grid_search.cv_results_['std_test_score'][best_idx]
    best_params = grid_search.best_params_

    print(f"GridSearchCV Complete!")
    print(f"  - Optimal Parameters: {best_params}")
    print(f"  - Tuned Model CV Mean Accuracy: {best_mean_acc:.4f} +/- {best_std_acc:.4f}")

    best_estimator = grid_search.best_estimator_

    # Evaluate tuned model on holdout test set
    X_test = exp_data['X_test_time']
    y_test = exp_data['y_test_time']
    y_pred_tuned = best_estimator.predict(X_test)
    y_prob_tuned = best_estimator.predict_proba(X_test)[:, 1]

    tuned_acc = accuracy_score(y_test, y_pred_tuned)
    tuned_auc = roc_auc_score(y_test, y_prob_tuned)
    print(f"  - Tuned Model Holdout Test Accuracy: {tuned_acc:.4f} | ROC-AUC: {tuned_auc:.4f}")

    return {
        'cv_summary': cv_summary,
        'best_estimator': best_estimator,
        'best_params': best_params,
        'tuned_cv_mean': float(best_mean_acc),
        'tuned_cv_std': float(best_std_acc),
        'tuned_test_acc': float(tuned_acc),
        'tuned_test_auc': float(tuned_auc)
    }

def generate_evaluation_visuals(exp_data, tuning_data):
    """
    Create and save high-resolution evaluation figures to results/:
    1. results/confusion_matrix.png
    2. results/roc_curve.png
    3. results/feature_importance.png
    """
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING EVALUATION VISUALS")
    print("=" * 70)
    os.makedirs('results', exist_ok=True)
    
    y_test = exp_data['y_test_time']
    best_pipe = tuning_data['best_estimator']
    y_pred = best_pipe.predict(exp_data['X_test_time'])
    y_prob = best_pipe.predict_proba(exp_data['X_test_time'])[:, 1]

    # --- 1. Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
        xticklabels=['Team B Wins', 'Team A Wins'],
        yticklabels=['Team B Wins', 'Team A Wins'],
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.72, f"({cm_norm[i, j]*100:.1f}%)", ha="center", va="center", color="gray", fontsize=10)
            
    ax.set_title("IPL Match Winner Prediction - Confusion Matrix\n(Tuned Random Forest on Time-Based Holdout)", fontsize=13, weight='bold', pad=15)
    ax.set_xlabel("Predicted Outcome", fontsize=11, weight='bold')
    ax.set_ylabel("Actual Outcome", fontsize=11, weight='bold')
    plt.tight_layout()
    cm_path = 'results/confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved: {cm_path}")

    # --- 2. ROC Curves Comparison ---
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    colors = {'Logistic Regression': '#1f77b4', 'Random Forest': '#2ca02c', 'Gradient Boosting': '#ff7f0e'}
    
    for name, preds in exp_data['predictions'].items():
        fpr, tpr, _ = roc_curve(y_test, preds['y_prob'])
        auc = roc_auc_score(y_test, preds['y_prob'])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=colors.get(name, 'blue'), lw=2)

    # Also plot tuned best model
    fpr_t, tpr_t, _ = roc_curve(y_test, y_prob)
    auc_t = roc_auc_score(y_test, y_prob)
    ax.plot(fpr_t, tpr_t, label=f"Tuned Random Forest (AUC = {auc_t:.3f})", color='#9467bd', lw=2.5, ls='--')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Baseline (AUC = 0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, weight='bold')
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, weight='bold')
    ax.set_title("ROC Curves - Model Comparison on Holdout Test Set", fontsize=13, weight='bold', pad=15)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    roc_path = 'results/roc_curve.png'
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved: {roc_path}")

    # --- 3. Feature Importance Plot ---
    # Extract feature names from preprocessor
    prep = best_pipe.named_steps['prep']
    num_names = exp_data['num_cols']
    cat_names = list(prep.named_transformers_['cat'].get_feature_names_out(exp_data['cat_cols']))
    all_feature_names = num_names + cat_names

    rf_clf = best_pipe.named_steps['clf']
    importances = rf_clf.feature_importances_

    # Map importances back to original features / top features
    feat_imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
    
    # Aggregate one-hot back to high-level features for clean presentation
    high_level = {}
    for f in exp_data['num_cols']:
        high_level[f] = feat_imp_df.loc[feat_imp_df['Feature'] == f, 'Importance'].sum()
    for cat in exp_data['cat_cols']:
        high_level[f"{cat} (One-Hot)"] = feat_imp_df.loc[feat_imp_df['Feature'].str.startswith(cat), 'Importance'].sum()
        
    hl_df = pd.DataFrame(list(high_level.items()), columns=['Feature', 'Importance'])
    hl_df = hl_df.sort_values(by='Importance', ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    bars = ax.barh(hl_df['Feature'], hl_df['Importance'], color='#1e3d59', edgecolor='black', alpha=0.85)
    ax.set_title("Pre-Match Feature Importance (Tuned Random Forest)", fontsize=13, weight='bold', pad=15)
    ax.set_xlabel("Relative Importance Weight", fontsize=11, weight='bold')
    for bar in bars:
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.3f}", va='center', fontsize=9, weight='bold')
    ax.set_xlim(0, hl_df['Importance'].max() * 1.15)
    plt.tight_layout()
    fi_path = 'results/feature_importance.png'
    plt.savefig(fi_path)
    plt.close()
    print(f"Saved: {fi_path}")

def run_team_clustering(match_df):
    """
    Perform KMeans clustering on team-level aggregates (win rate, avg runs, avg wickets).
    Evaluate elbow (inertia) and silhouette score curves across k in [2, 6].
    Save results/kmeans_elbow_silhouette.png and results/team_clusters.png.
    """
    print("\n" + "=" * 70)
    print("STEP 7: TEAM PLAYING STYLE CLUSTERING (KMEANS)")
    print("=" * 70)
    os.makedirs('results', exist_ok=True)
    
    team_stats = []
    teams = set(match_df['Team_A']).union(set(match_df['Team_B']))

    for t in teams:
        matches = match_df[(match_df['Team_A'] == t) | (match_df['Team_B'] == t)]
        n_matches = len(matches)
        if n_matches < 10:
            continue
        wins = (match_df['Winner'] == t).sum()
        win_rate = wins / n_matches
        avg_runs = matches['total_runs'].mean()
        avg_wickets = matches['total_wickets'].mean()
        team_stats.append({
            'Team': t,
            'Matches': n_matches,
            'Wins': wins,
            'Win_Rate': win_rate,
            'Avg_Runs': avg_runs,
            'Avg_Wickets': avg_wickets
        })

    t_df = pd.DataFrame(team_stats)
    X = t_df[['Win_Rate', 'Avg_Runs', 'Avg_Wickets']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_range = range(2, 6)
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        print(f"  k={k}: Inertia = {km.inertia_:.4f} | Silhouette Score = {silhouettes[-1]:.4f}")

    # Plot Elbow & Silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    ax1.plot(k_range, inertias, 'bo-', lw=2, markersize=8)
    ax1.set_title("Elbow Method (Inertia vs. k)", fontsize=12, weight='bold')
    ax1.set_xlabel("Number of Clusters (k)", fontsize=10, weight='bold')
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=10, weight='bold')

    ax2.plot(k_range, silhouettes, 'rs-', lw=2, markersize=8)
    ax2.set_title("Silhouette Score vs. k", fontsize=12, weight='bold')
    ax2.set_xlabel("Number of Clusters (k)", fontsize=10, weight='bold')
    ax2.set_ylabel("Average Silhouette Coefficient", fontsize=10, weight='bold')
    plt.tight_layout()
    clust_eval_path = 'results/kmeans_elbow_silhouette.png'
    plt.savefig(clust_eval_path)
    plt.close()
    print(f"Saved: {clust_eval_path}")

    # Choose k=3 as optimal balance of parsimony and distinct playing styles
    chosen_k = 3
    final_km = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    t_df['Cluster'] = final_km.fit_predict(X_scaled)

    # Cluster visualization
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    palette = ['#ff6e40', '#1e3d59', '#2ecc71']
    sns.scatterplot(
        data=t_df, x='Avg_Runs', y='Win_Rate', hue='Cluster',
        palette=palette, s=160, edgecolor='black', alpha=0.9, ax=ax
    )
    for idx, r in t_df.iterrows():
        # Clean label for plot
        label = r['Team'].replace('Royal Challengers Bengaluru', 'RCB').replace('Chennai Super Kings', 'CSK').replace('Mumbai Indians', 'MI').replace('Kolkata Knight Riders', 'KKR').replace('Sunrisers Hyderabad', 'SRH').replace('Rajasthan Royals', 'RR').replace('Delhi Capitals', 'DC').replace('Punjab Kings', 'PBKS').replace('Gujarat Titans', 'GT').replace('Lucknow Super Giants', 'LSG')
        ax.annotate(label, (r['Avg_Runs'] + 0.8, r['Win_Rate']), fontsize=9, weight='bold')
        
    ax.set_title(f"IPL Team Clusters by Playing Style (k={chosen_k})", fontsize=13, weight='bold', pad=15)
    ax.set_xlabel("Average Match Total Runs (Tempo/Conditions)", fontsize=11, weight='bold')
    ax.set_ylabel("Historical Win Rate", fontsize=11, weight='bold')
    plt.tight_layout()
    clust_vis_path = 'results/team_clusters.png'
    plt.savefig(clust_vis_path)
    plt.close()
    print(f"Saved: {clust_vis_path}")

    return {
        'team_clusters_df': t_df,
        'inertias': inertias,
        'silhouettes': silhouettes
    }

def save_artifacts(best_model, match_df):
    """
    Export all production artifacts needed by app.py and downstream consumers.
    """
    print("\n" + "=" * 70)
    print("STEP 8: SAVING PRODUCTION ARTIFACTS")
    print("=" * 70)
    
    # 1. Save models (best_model.pkl and rf_model.pkl for seamless backward compatibility)
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(best_model, 'rf_model.pkl')
    print("Saved 'best_model.pkl' and updated 'rf_model.pkl'.")

    # 2. Save encoders and team/venue registry
    all_teams = sorted(list(set(match_df['Team_A']).union(set(match_df['Team_B']))))
    all_venues = sorted(match_df['Venue'].unique().tolist())
    toss_decisions = sorted(match_df['Toss_Decision'].unique().tolist())

    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    team_le = LabelEncoder()
    team_le.fit(all_teams)
    for col in ['Team_A', 'Team_B', 'Toss_Winner', 'Winner']:
        encoders[col] = team_le

    venue_le = LabelEncoder()
    venue_le.fit(all_venues)
    encoders['Venue'] = venue_le

    toss_le = LabelEncoder()
    toss_le.fit(toss_decisions)
    encoders['Toss_Decision'] = toss_le

    joblib.dump(encoders, 'encoders.pkl')
    print("Saved 'encoders.pkl' registry.")

    # 3. Save clean decoded match history
    match_df.to_pickle('match_history.pkl')
    match_df.to_csv('match_history.csv', index=False)
    print("Saved 'match_history.pkl' and 'match_history.csv'.")

def main():
    print("=" * 70)
    print("STARTING IPL ML PIPELINE TRAINING & EVALUATION")
    print("=" * 70)
    
    # 1. Load data
    match_df = load_and_preprocess_data()

    # 2. Leakage-free features
    single_df, sym_df = build_leakage_free_features(match_df)

    # 3. Baselines
    baselines = compute_baselines(match_df, test_season_threshold=2024)

    # 4. Model comparison (Time split & Stratified split)
    exp_data = run_model_experiments(sym_df, test_season_threshold=2024)

    # 5. Cross validation & Hyperparameter tuning
    tuning_data = run_cross_validation_and_tuning(exp_data)

    # 6. Evaluation visuals
    generate_evaluation_visuals(exp_data, tuning_data)

    # 7. Clustering
    run_team_clustering(match_df)

    # 8. Save artifacts
    save_artifacts(tuning_data['best_estimator'], match_df)

    # Save summary metadata JSON for dashboard display
    summary = {
        'train_test_split': {
            'method': 'Time-Based Split (Train: Seasons 2008-2023, Test: Seasons 2024-2026)',
            'train_matches': int((match_df['Season'] < 2024).sum()),
            'test_matches': int((match_df['Season'] >= 2024).sum()),
            'train_symmetric_rows': len(exp_data['X_train_time']),
            'test_symmetric_rows': len(exp_data['X_test_time']),
            'stratified_split': '80% Train, 20% Test (Stratified on Match Outcome)'
        },
        'baselines': baselines,
        'cv_scores': tuning_data['cv_summary'],
        'tuned_cv': {
            'mean_accuracy': tuning_data['tuned_cv_mean'],
            'std_accuracy': tuning_data['tuned_cv_std'],
            'best_params': tuning_data['best_params'],
            'holdout_test_accuracy': tuning_data['tuned_test_acc'],
            'holdout_test_auc': tuning_data['tuned_test_auc']
        }
    }
    with open('results/model_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE! ALL ARTIFACTS SAVED SUCCESSFULLY.")
    print("=" * 70)

if __name__ == "__main__":
    main()
