import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    # Load the ball-by-ball dataset
    df = pd.read_csv('IPL.csv', low_memory=False)

    print("Aggregating ball-by-ball data...")
    # Squash ball-by-ball data to match level
    agg_df = df.groupby('match_id').agg(
        total_runs=('runs_total', 'sum'),
        total_wickets=('player_out', lambda x: x.notna().sum()),
        Venue=('venue', 'first'),
        Toss_Winner=('toss_winner', 'first'),
        Toss_Decision=('toss_decision', 'first'),
        Team_A=('batting_team', 'first'),
        Team_B=('bowling_team', 'first'),
        Winner=('match_won_by', 'first')
    ).reset_index()

    # Drop any matches with missing target or feature values
    agg_df = agg_df.dropna(subset=['Winner', 'Venue', 'Toss_Winner', 'Toss_Decision', 'Team_A', 'Team_B'])

    print("Encoding features...")
    encoders = {}
    
    # Create a single label encoder for all team names to ensure consistency
    team_encoder = LabelEncoder()
    all_teams = pd.concat([agg_df['Team_A'], agg_df['Team_B'], agg_df['Toss_Winner'], agg_df['Winner']]).astype(str).unique()
    team_encoder.fit(all_teams)
    
    for col in ['Team_A', 'Team_B', 'Toss_Winner', 'Winner']:
        agg_df[col] = team_encoder.transform(agg_df[col].astype(str))
        encoders[col] = team_encoder

    # Create encoders for Venue and Toss Decision
    for col in ['Venue', 'Toss_Decision']:
        le = LabelEncoder()
        agg_df[col] = le.fit_transform(agg_df[col].astype(str))
        encoders[col] = le

    # Feature selection
    features = ['Venue', 'Toss_Winner', 'Toss_Decision', 'Team_A', 'Team_B']
    X = agg_df[features]
    y = agg_df['Winner']

    print("Training Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    print("Saving model and encoders...")
    joblib.dump(rf, 'rf_model.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    
    # Save match history for dashboard stats
    agg_df.to_pickle('match_history.pkl')
    
    print("Training complete! Model (rf_model.pkl), encoders (encoders.pkl), and history (match_history.pkl) saved.")

if __name__ == "__main__":
    main()
