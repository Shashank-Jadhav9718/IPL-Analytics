# IPL Match Winner Predictor 🏏

This project is a Machine Learning pipeline and Streamlit dashboard that predicts the outcome of Indian Premier League (IPL) matches. It uses historical ball-by-ball data to train a Random Forest classifier and provides a clean, modern UI for users to generate predictions with detailed statistical justifications.

## Features ✨

- **Machine Learning Model**: A Random Forest Classifier trained on historical IPL match data.
- **Modern Dashboard**: A clean, Glassmorphism-themed Streamlit application.
- **Detailed Justifications**: The dashboard doesn't just predict the winner; it explains *why* using:
  - ⚔️ **Head-to-Head**: Historical win ratios and recent encounters between the selected teams.
  - 🏟️ **Venue Analysis**: Top-performing teams and the impact of the toss at the selected venue.
  - 🤖 **Model Insights**: Feature importance charts showing what the algorithm prioritized.
  - 📝 **Final Conclusion**: A dynamic, written rationale summarizing the prediction drivers.
- **Constrained Selections**: Dropdowns are intelligently filtered to only show the 10 currently active IPL franchises to ensure relevant predictions.

## Dataset 📊

The model is trained on a comprehensive ball-by-ball dataset of historical IPL matches (`IPL.csv`). The dataset contains rich information including:
- Match details (teams, venue, toss winner, toss decision, match winner).
- Ball-by-ball events (runs scored, wickets taken, extras, etc.).
- The data is aggregated from a ball-by-ball level into match-level summaries before training the prediction model.

*Note: Due to file size limits, the `IPL.csv` dataset is excluded from this repository and must be obtained separately.*

## Project Structure 📁

- `train_model.py`: The data aggregation and model training script. It processes the raw ball-by-ball dataset (`IPL.csv`), trains the Random Forest model, and exports the necessary artifacts (`rf_model.pkl`, `encoders.pkl`, `match_history.pkl`).
- `app.py`: The Streamlit dashboard application.

## Installation & Setup 🚀

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Install dependencies:**
   Ensure you have Python installed, then install the required packages:
   ```bash
   pip install pandas scikit-learn streamlit plotly joblib
   ```

3. **Data Preparation:**
   Due to GitHub file size limits, the raw `IPL.csv` dataset is not included in this repository. 
   - Place your `IPL.csv` (ball-by-ball dataset) in the root directory.
   - Run the training script to generate the models:
     ```bash
     python train_model.py
     ```

4. **Run the Application:**
   Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

## Requirements 📦
- Python 3.8+
- pandas
- scikit-learn
- streamlit
- plotly
- joblib

---
*Built with Python and Streamlit.*
