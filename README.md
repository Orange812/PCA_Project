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

The system's database is built from a comprehensive set of leagues, including major European and South American competitions. The dynamic data loader automatically discovers and processes all available data. As of the latest update, the following leagues are included:

- **Europe:**
  - Austria: Bundesliga
  - Belgium: Pro League
  - Croatia: Prva HNL
  - Czech Republic: First League
  - Denmark: Superliga
  - England: Premier League, Championship
  - France: Ligue 1, Ligue 2
  - Germany: Bundesliga, 2. Bundesliga, Play-offs 1-2
  - Greece: Super League
  - Italy: Serie A, Serie B
  - Netherlands: Eredivisie
  - Norway: Eliteserien
  - Poland: Ekstraklasa
  - Portugal: Liga NOS, Ligapro
  - Russia: Russian Premier League
  - Scotland: Premiership
  - Serbia: Superliga
  - Spain: La Liga, Segunda Division
  - Sweden: Allsvenskan
  - Switzerland: Super League
  - Turkey: Super Lig
- **South America:**
  - Brazil: Serie A

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

## Feature Engineering

The model's performance heavily relies on a set of carefully crafted features that capture team form, style, and market sentiment. The corner prediction model, for example, uses the following key features.

### Key Features and Their Meanings

| Feature Name            | Real-world Meaning                                                                                                                              |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `home_avg_possession`   | The home team's average possession. Reflects the team's ability to control the game's tempo and dominate the attack.                                |
| `home_avg_shots`        | The home team's average number of shots. Directly correlates to the frequency of creating scoring opportunities.                                |
| `home_avg_corners`      | The home team's average number of corners. Often indicates the team's activity in wide attacks and its ability to apply continuous pressure.         |
| `home_avg_xg`           | The home team's average expected goals (xG). Quantifies the quality of each shot and is a core metric for offensive efficiency.                      |
| `away_avg_possession`   | The away team's average possession.                                                                                                             |
| `away_avg_shots`        | The away team's average number of shots.                                                                                                        |
| `away_avg_corners`      | The away team's average number of corners.                                                                                                      |
| `away_avg_xg`           | The away team's average expected goals (xG).                                                                                                    |
| `home_win_fair_prob`    | The fair probability of a home win implied by market odds. Represents the betting market's consensus on the home team's likelihood of winning.      |
| `strength_disparity`    | Strength disparity (fair prob. of home win - fair prob. of away win). Quantifies the market's perceived difference in strength between the two teams. |
| `over_25_prob`          | The probability of total goals exceeding 2.5, as implied by market odds. Reflects the market's overall judgment on the match's scoring potential. |
| `btts_yes_prob`         | The probability that both teams will score, as implied by market odds.                                                                          |

### Example Feature Correlation Matrix

During development, analyzing the correlation between different features is crucial to avoid multicollinearity and understand feature relationships. The following is a correlation matrix for a set of defensive ratio features explored in an early-stage model. Values close to 1 or -1 indicate a strong correlation.

```
           ratio1    ratio2    ratio3    ratio4    ratio5    ratio6
ratio1  1.000000  0.510050  0.034400  0.157528  0.033620  0.706218
ratio2  0.510050  1.000000  0.020582  0.273658  0.019180  0.696237
ratio3  0.034400  0.020582  1.000000  0.019620  0.003637  0.037109
ratio4  0.157528  0.273658  0.019620  1.000000 -0.078062  0.357276
ratio5  0.033620  0.019180  0.003637 -0.078062  1.000000  0.009745
ratio6  0.706218  0.696237  0.037109  0.357276  0.009745  1.000000
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
- [x] Develop a web interface
- [ ] Support more betting markets

# Corner Prediction Web UI (Feature Branch)

This project now includes a web-based user interface for the Corner Prediction Model, running on a separate feature branch. This allows users to easily get predictions by inputting pre-match statistics for two competing teams without running command-line scripts.

The application is built with Flask and styled with Bootstrap for a clean, modern, and responsive user experience.

## How to Run the Web UI

1.  **Ensure you are on the correct branch**:
    ```bash
    git checkout feature/corner-prediction-ui
    ```

2.  **Install dependencies**:
    Make sure you have all the required Python packages installed, including the newly added `scipy`.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    Execute the `web_app.py` script.
    ```bash
    python web_app.py
    ```

4.  **Access in browser**:
    Open your web browser and navigate to the following address:
    [http://127.0.0.1:5001/](http://127.0.0.1:5001/)

## Prediction Example

Below is an example prediction for a match between a strong home team and a weaker away team.

### Input Data:

| Feature                 | Home Team | Away Team |
| ----------------------- | :-------: | :-------: |
| Pre-Match PPG           |   2.10    |   1.20    |
| Average Shots           |   15.0    |   8.0     |
| Average Shots on Target |   6.0     |   3.0     |
| Pre-Match xG            |   2.20    |   1.10    |
| Average Possession (%%)  |   60.0    |   40.0    |
| Average Corners For     |   7.0     |   4.0     |
| Average Corners Against |   3.0     |   6.0     |

### Prediction Output:

*   **Home Team Expected Corners**: 7.55
*   **Away Team Expected Corners**: 4.10
*   **Total Expected Corners**: 11.65

#### Total Corners Probability Distribution

| Total Corners | Probability |
| :-----------: | :---------- |
|       7       | 5.61%%       |
|       8       | 8.14%%       |
|       9       | 10.45%%      |
|      10       | 11.97%%      |
|      **11**     | **12.41%%**  |
|      12       | 11.74%%      |
|      13       | 10.19%%      |
|      14       | 8.15%%       |
|      15       | 6.02%%       |
|      16       | 4.13%%       |

*(Probabilities for other totals are calculated but not shown for brevity)*

#### Home/Away Corners Poisson Distribution Matrix (%%)

This matrix shows the probability of each specific corner outcome (Home x Away).

| Away↓ / Home→ | 5      | 6      | 7      | **8**  | 9      | 10     |
| :------------ | :----- | :----- | :----- | :----- | :----- | :----- |
| 2             | 1.28%%  | 1.61%%  | 1.73%%  | 1.63%%  | 1.37%%  | 1.04%%  |
| 3             | 1.76%%  | 2.22%%  | 2.38%%  | 2.24%%  | 1.88%%  | 1.43%%  |
| **4**         | **1.72%%** | **2.16%%** | **2.32%%** | **2.18%%** | **1.83%%** | **1.39%%** |
| 5             | 1.41%%  | 1.77%%  | 1.90%%  | 1.79%%  | 1.50%%  | 1.14%%  |
| 6             | 0.97%%  | 1.22%%  | 1.31%%  | 1.23%%  | 1.03%%  | 0.78%%  |

*(Matrix is truncated for display purposes)*


## Author

Football Betting Prediction and Analysis System - An intelligent prediction platform based on ELO ratings and machine learning.

---

*This project demonstrates how to combine a traditional sports rating system with modern machine learning techniques to create a practical football match prediction system.*