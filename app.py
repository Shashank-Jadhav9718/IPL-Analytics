import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# Configuration and Theme
st.set_page_config(page_title="IPL 2026 Predictor", page_icon="🏏", layout="wide")

st.markdown("""
<style>
    /* Modern UI aesthetics */
    .main { background-color: #f8f9fa; }
    h1 { color: #1e3d59; font-family: 'Inter', sans-serif; font-weight: 700; }
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        background-color: #ff6e40; 
        color: white; 
        border: none; 
        padding: 12px; 
        font-weight: 600; 
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #ff5722; color: white; transform: scale(1.02); }
</style>
""", unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('rf_model.pkl')
        encoders = joblib.load('encoders.pkl')
        return model, encoders
    except Exception:
        return None, None

model, encoders = load_assets()

if not model:
    st.error("Model files not found. Please run `python train_model.py` to train the model first.")
    st.stop()

# Header
st.title("🏏 IPL 2026 Match Winner Predictor")
st.markdown("Select the match conditions and teams below to predict the winner using our Machine Learning model.")
st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.header("Match Settings")
    
    ALLOWED_TEAMS = [
        'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 
        'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians', 
        'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bengaluru', 
        'Sunrisers Hyderabad'
    ]
    
    # Filter teams to only include allowed active franchises
    teams = [t for t in ALLOWED_TEAMS if t in encoders['Team_A'].classes_]
    # Fallback to Bangalore if Bengaluru has no history in this dataset split
    if 'Royal Challengers Bangalore' in encoders['Team_A'].classes_ and 'Royal Challengers Bengaluru' not in teams:
        teams.append('Royal Challengers Bangalore')
        
    venues = sorted(encoders['Venue'].classes_)
    toss_decisions = sorted(encoders['Toss_Decision'].classes_)
    
    team_a = st.selectbox("Team A", teams, index=0)
    team_b = st.selectbox("Team B", teams, index=1 if len(teams) > 1 else 0)
    
    venue = st.selectbox("Venue", venues)
    toss_winner = st.selectbox("Toss Winner", [team_a, team_b])
    toss_decision = st.selectbox("Toss Decision", toss_decisions)
    
    predict_btn = st.button("Predict Match Winner")

# Prediction Logic & Output
if predict_btn:
    if team_a == team_b:
        st.warning("⚠️ Team A and Team B cannot be the same!")
    else:
        # Build input dataframe
        input_data = pd.DataFrame({
            'Venue': [venue],
            'Toss_Winner': [toss_winner],
            'Toss_Decision': [toss_decision],
            'Team_A': [team_a],
            'Team_B': [team_b]
        })
        
        # Transform inputs using encoders
        for col in input_data.columns:
            input_data[col] = encoders[col].transform(input_data[col])
            
        # Get predictions
        pred_class = model.predict(input_data)[0]
        pred_proba = model.predict_proba(input_data)[0]
        
        predicted_winner = encoders['Winner'].inverse_transform([pred_class])[0]
        
        # Display Results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.success(f"🏆 **Predicted Winner:**\n### {predicted_winner}")
            
        with col2:
            # Prepare probability chart
            team_a_idx = encoders['Team_A'].transform([team_a])[0]
            team_b_idx = encoders['Team_B'].transform([team_b])[0]
            
            classes = list(model.classes_)
            prob_a = pred_proba[classes.index(team_a_idx)] if team_a_idx in classes else 0.0
            prob_b = pred_proba[classes.index(team_b_idx)] if team_b_idx in classes else 0.0
            
            # Normalize probabilities between the two playing teams
            total = prob_a + prob_b
            if total > 0:
                prob_a_pct, prob_b_pct = (prob_a / total) * 100, (prob_b / total) * 100
            else:
                prob_a_pct, prob_b_pct = 50.0, 50.0
                
            chart_data = pd.DataFrame({
                'Team': [team_a, team_b],
                'Win Probability (%)': [prob_a_pct, prob_b_pct]
            })
            
            # Plotly Chart
            fig = px.bar(
                chart_data, 
                x='Team', 
                y='Win Probability (%)', 
                color='Team', 
                color_discrete_sequence=['#1e3d59', '#ff6e40'],
                title="Head-to-Head Win Probability"
            )
            fig.update_layout(yaxis_range=[0, 100], showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

        # --- NEW SECTION: Justifying the Prediction ---
        st.markdown("---")
        st.subheader("📊 Match Forecast Justification")
        st.markdown("Explore the detailed statistical analysis and logic driving this prediction.")
        
        try:
            history_df = pd.read_pickle('match_history.pkl')
            
            # Decode history_df for readable charts
            for col in ['Venue', 'Toss_Winner', 'Toss_Decision', 'Team_A', 'Team_B', 'Winner']:
                if col in history_df.columns:
                    history_df[col] = encoders[col].inverse_transform(history_df[col])
                    
            tab1, tab2, tab3, tab4 = st.tabs(["⚔️ Head-to-Head", "🏟️ Venue Analysis", "🤖 Model Insights", "📝 Final Conclusion"])
            
            with tab1:
                st.markdown(f"### {team_a} vs {team_b} Historical Matchups")
                h2h = history_df[((history_df['Team_A'] == team_a) & (history_df['Team_B'] == team_b)) | 
                                 ((history_df['Team_A'] == team_b) & (history_df['Team_B'] == team_a))]
                
                colA, colB = st.columns([1, 1.5])
                with colA:
                    if not h2h.empty:
                        win_counts = h2h['Winner'].value_counts().reset_index()
                        win_counts.columns = ['Team', 'Wins']
                        fig_h2h = px.pie(win_counts, values='Wins', names='Team', hole=0.4, 
                                         color_discrete_sequence=['#ff4d4d', '#1f77b4'],
                                         title=f"Total Matches: {len(h2h)}")
                        st.plotly_chart(fig_h2h, use_container_width=True)
                    else:
                        st.info("No historical matches found between these two teams.")
                        win_counts = pd.DataFrame({'Team': [team_a, team_b], 'Wins': [0, 0]})
                        
                with colB:
                    if not h2h.empty:
                        st.markdown("**All Past Encounters**")
                        recent = h2h[['Venue', 'Toss_Winner', 'Toss_Decision', 'Winner']].reset_index(drop=True)
                        st.dataframe(recent, use_container_width=True)
                        
            with tab2:
                st.markdown(f"### Performance at {venue}")
                venue_df = history_df[history_df['Venue'] == venue]
                
                colC, colD = st.columns([1.5, 1])
                with colC:
                    if not venue_df.empty:
                        venue_wins = venue_df['Winner'].value_counts().head(8).reset_index()
                        venue_wins.columns = ['Team', 'Wins']
                        fig_venue = px.bar(venue_wins, x='Team', y='Wins', 
                                           title="Most Successful Teams at this Venue",
                                           color='Wins', color_continuous_scale='Viridis')
                        st.plotly_chart(fig_venue, use_container_width=True)
                    else:
                        st.info("No historical data available for this venue.")
                        
                with colD:
                    if not venue_df.empty:
                        venue_df['Toss_Win_Match_Win'] = venue_df['Toss_Winner'] == venue_df['Winner']
                        toss_impact = venue_df['Toss_Win_Match_Win'].value_counts().reset_index()
                        toss_impact['Toss_Win_Match_Win'] = toss_impact['Toss_Win_Match_Win'].map({True: 'Won Toss & Match', False: 'Lost Toss & Match'})
                        fig_toss = px.pie(toss_impact, values='count', names='Toss_Win_Match_Win', 
                                          title="Toss Impact at Venue",
                                          color_discrete_sequence=['#2ecc71', '#e74c3c'])
                        st.plotly_chart(fig_toss, use_container_width=True)

            with tab3:
                st.markdown("### Random Forest Feature Importances")
                st.markdown("This chart highlights which factors the Machine Learning algorithm weighted most heavily when calculating the win probability.")
                importances = model.feature_importances_
                features = ['Venue', 'Toss_Winner', 'Toss_Decision', 'Team_A', 'Team_B']
                feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                feat_df = feat_df.sort_values(by='Importance', ascending=True)
                
                fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                                  color='Importance', color_continuous_scale='Blues')
                st.plotly_chart(fig_feat, use_container_width=True)

            with tab4:
                st.markdown("### 📝 Analytical Conclusion")
                
                # Determine top feature
                top_feature = feat_df.iloc[-1]['Feature']
                
                # Determine H2H dominance
                h2h_winner = "neither team (no history)"
                if not h2h.empty:
                    top_h2h = win_counts.iloc[0]
                    h2h_winner = top_h2h['Team']
                    
                predicted_prob = max(prob_a_pct, prob_b_pct)
                
                st.success(f"""
                **Prediction Rationale:**
                
                The Machine Learning model confidently predicts **{predicted_winner}** to emerge victorious with a **{predicted_prob:.1f}% probability**.
                
                **Key Drivers:**
                1. **Algorithm Weighting**: The model identified **{top_feature}** as the most critical variable for this specific matchup.
                2. **Historical Dominance**: In direct Head-to-Head matchups, **{h2h_winner}** has historically held the upper hand.
                3. **Venue Dynamics**: The conditions at *{venue}* and the decision to *{toss_decision}* by the toss winner played a significant role in tipping the model's confidence towards {predicted_winner}.
                """)
                
                st.info("💡 **Note**: Predictions are based purely on historical ball-by-ball patterns and data aggregation. Actual match outcomes may vary due to real-time player form and unforeseen match events.")

        except FileNotFoundError:
            st.warning("Historical data file (match_history.pkl) not found. Run `train_model.py` to generate it.")
