# Football Betting Prediction and Analysis System

## System Overview

This is a football betting prediction and analysis system based on ELO ratings and machine learning. It can predict the number of goals in a match and provide betting advice based on the recent goal-scoring and goal-conceding performance of the teams.

## Core Features

### 🏆 ELO Rating System
- **Traditional ELO Algorithm**: Quantifies team strength.
- **Home and Away Advantage**: Calculates a constant home and away advantage coefficient (0.070) based on historical data.
- **ELO Inheritance System**: Each season, teams inherit the ELO score from the previous season.
- **Promotion and Relegation Handling**: Promoted teams get the league's lowest ELO, and relegated teams get the league's highest ELO.

### ⚽ Goal Prediction Algorithm
- **ELO-Weighted Prediction**: Adjusts the weight of historical goals based on the strength difference of opponents.
- **Optimal Historical Matches**: Determines the optimal number of historical matches (13) through cross-validation.
- **Time Decay**: More recent matches have a higher weight.
- **Confidence Assessment**: Quantifies the reliability of the predictions.

### 🧹 Data Cleaning
- **Outlier Match Filtering**: Automatically removes matches with a goal difference greater than 4.
- **Data Standardization**: Standardizes team name formats.
- **Validity Verification**: Ensures the integrity of match data.

### 🤖 Machine Learning Enhancement
- **Random Forest Model**: Independent models are trained for the home and away teams.
- **Feature Engineering**: Extracts 17 key features.
- **Model Fusion**: A weighted combination of ELO predictions and machine learning predictions.

### 🔮 Corner Prediction Model (XGBoost)
- **Objective**: Predicts the total number of corners in a match using a Poisson regression model powered by XGBoost.
- **Features**: The model utilizes a rich set of contextual features, including:
    - **Team-specific stats**: Average possession, shots, corners, and expected goals (xG) for both home and away scenarios.
    - **Betting Odds**: Implied probabilities from betting odds are used to gauge market sentiment and team strength disparity (e.g., home win probability, over 2.5 goals probability).
- **Data Handling**: The system intelligently maps team names, processes data from multiple seasons and leagues, and creates a feature set that respects the context of home and away performance.

## System Architecture

```
/ (Project Root)
├── src/
│   └── football_predictor/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py             # Data loading and preprocessing
│       ├── features/
│       │   ├── __init__.py
│       │   └── engineering.py        # Feature engineering functions
│       ├── models/
│       │   ├── __init__.py
│       │   ├── elo_system.py         # Core ELO rating system
│       │   ├── goal_predictor.py     # Advanced ML-based goal prediction
│       │   ├── corner_predictor.py   # XGBoost corner prediction model
│       │   └── adaptive_system.py    # Adaptive weighting and betting strategy
│       ├── utils/
│       │   ├── __init__.py
│       │   └── config.py             # Project configuration
│       └── betting_advisor.py        # Comprehensive betting advice generator
├── notebooks/                          # Jupyter notebooks for analysis and exploration
│   ├── Adaboost.ipynb
│   ├── demo.ipynb
│   └── ...
├── data/                               # Raw CSV data (unchanged)
│   └── ...
├── main.py                             # Main entry point for the application
├── requirements.txt                    # Project dependencies
└── README.md                           # This file
```

## Data Sources

The system uses match data from major European leagues:
- Premier League (England)
- Bundesliga (Germany)
- La Liga (Spain)
- Ligue 1 (France)
- Serie A (Italy)
- Eredivisie (Netherlands)
- Liga NOS (Portugal)
- Superliga (Denmark)
- Championship (England)
- Serie B (Italy)
- 2. Bundesliga (Germany)

Timeframe: 2020-2024 seasons

## Core Algorithms

### 1. Home and Away Advantage Calculation
```python
home_advantage = (home_win_rate - away_win_rate) / 2
# Result: 0.070 (7% home advantage)
```

### 2. ELO Update Formula
```python
expected_home = 1 / (1 + 10**((away_elo - home_elo - home_advantage*400) / 400))
new_elo = old_elo + K * (actual_result - expected_result)
```

### 3. ELO-Weighted Goal Prediction
```python
weight = time_decay_factor * exp(-|opponent_elo_diff|**2 / (2 * sigma**2))
predicted_goals = sum(goals_i * weight_i) / sum(weight_i)
```

## Usage

This project now uses a centralized `main.py` script as its entry point.

### 1. Initialize and Train the System
First, you need to run the initialization process. This will process all the data, train the necessary models, and find the optimal weights.

```bash
python main.py init
```

### 2. Make a Prediction
Once the system is initialized, you can make predictions for a specific match.

```bash
python main.py predict --home "manchester city" --away "arsenal"
```

## Prediction Result Examples

### Goal Prediction
```
🏟️ Premier League Clash: Manchester City vs Arsenal
📊 ELO Scores: 1589 vs 1584
⚽ Predicted Score: 1.5 - 2.3
🎯 Total Goals: 3.8
📈 Win/Draw/Loss Probability: Home Win 46.5% | Draw 30.0% | Away Win 38.5%
🔒 Confidence: 0.25
💡 Advice: Over 3.5 | Result uncertain
```

### Corner Prediction Model Performance

The following table shows the learning curve analysis for the corner prediction model. The Mean Absolute Error (MAE) is measured on a fixed test set as the size of the training dataset increases.

| Training Samples | MAE on Test Set |
|------------------|-----------------|
| 3514             | 2.6589          |
| 7029             | 2.6322          |
| 14058            | 2.6149          |
| 21087            | 2.6180          |
| 28116            | 2.6063          |
| 35145            | 2.6037          |

**Conclusion**: The model's performance steadily improves as the amount of data increases. The MAE decreased by 0.0552 from the smallest to the largest dataset. This clearly indicates that the model can still benefit from more data. It is recommended to add more seasons or leagues to further enhance the model's prediction accuracy.

## System Performance

### Data Processing Capability
- ✅ Processed data for 44 league seasons
- ✅ Automatically cleaned ~300 outlier matches
- ✅ Established a complete historical database of team ELOs

### Prediction Accuracy
- 📊 Optimal number of historical matches: 13
- 📊 Cross-validation accuracy: 43%
- 📊 Machine learning model MAE: 0.87 (Home), 0.81 (Away)

### Betting Advice Types
- 🎯 Total Goals (Over/Under)
- 🏆 Match Winner (1X2)
- ⚽ Both Teams to Score (BTTS)
- 📊 Confidence Assessment

## Technology Stack

- **Python 3.8+**
- **pandas** - Data processing
- **numpy** - Numerical computation
- **sqlite3** - Data storage
- **scikit-learn** - Machine learning
- **xgboost** - Gradient Boosting library

## Installation

```bash
pip install pandas numpy scikit-learn xgboost
```

## File Descriptions

### Core Modules
- `football_betting_system.py` - Main system, including ELO calculation and basic prediction.
- `advanced_goal_prediction.py` - Machine learning enhanced prediction module.
- `corner_prediction_xgboost.py` - Corner prediction model using XGBoost.
- `betting_advisor.py` - Betting advice generator.

### Demonstration Scripts
- `demo_system.py` - Full system demonstration.
- `ELO.ipynb` - Original ELO algorithm implementation.

### Data Files
- `data/` - Contains match and team data for various leagues.
- `football_betting.db` - SQLite database file.

## System Advantages

1.  **Scientific**: Quantitative analysis based on the ELO rating system.
2.  **Accurate**: Enhanced predictions with machine learning models.
3.  **Practical**: Direct output of betting advice.
4.  **Reliable**: Confidence assessment and risk control.
5.  **Complete**: Full pipeline from data processing to advice generation.

## Important Notes

⚠️ **Risk Warning**: This system is for learning and research purposes only and does not constitute investment advice. Betting is risky; wager with caution.

⚠️ **Data Dependency**: Prediction accuracy depends on the quality and completeness of historical data.

⚠️ **Model Limitations**: Football matches involve many unpredictable factors, and all prediction models have limitations.

## Future Improvements

- [ ] Add more league data
- [ ] Integrate real-time data sources
- [ ] Optimize machine learning models
- [ ] Add player injury and transfer information
- [ ] Develop a web interface
- [ ] Support more betting markets

## Author

Football Betting Prediction and Analysis System - An intelligent prediction platform based on ELO ratings and machine learning.

---

*This project demonstrates how to combine a traditional sports rating system with modern machine learning techniques to create a practical football match prediction system.*