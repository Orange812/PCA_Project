# feature_engineering.py
# 描述: 包含所有用于特征工程的函数，如防守统计、ELO评分等。

import numpy as np
import pandas as pd

# ====================== 计算防守统计数据 ======================
def compute_defensive_stats(match_df, team_positions_df):
    """计算球队的防守统计数据，包括6种比率、总失球数和平均失球数"""
    epsilon = 1e-8  # 提高数值稳定性
    team_stats = {}
    for team in team_positions_df['team_name'].unique():
        team_stats[team] = {
            'total_goals_conceded': 0, 'ratio1_list': [], 'ratio2_list': [], 'ratio3_list': [],
            'ratio4_list': [], 'ratio5_list': [], 'ratio6_list': [], 'num_matches': 0
        }

    # 主场比赛统计
    for idx, row in match_df.iterrows():
        home_team, away_team = row['home_team_name'], row['away_team_name']
        if home_team not in team_stats or away_team not in team_stats:
            print(f"警告: 主队 {home_team} 或客队 {away_team} 未找到，跳过比赛 {idx}")
            continue
        if row['Pre-Match PPG (Away)'] > 0:
            team_stats[home_team]['ratio1_list'].append(
                row['away_team_goal_count'] / (row['Pre-Match PPG (Away)'] + epsilon))
        if row['away_team_corner_count'] > 0:
            team_stats[home_team]['ratio2_list'].append(
                row['away_team_goal_count'] / (row['away_team_corner_count'] + epsilon))
        denominator = row['home_team_yellow_cards'] + row['home_team_red_cards'] + row['home_team_fouls'] + epsilon
        team_stats[home_team]['ratio3_list'].append(row['away_team_goal_count'] / denominator)
        away_xg = row['away_xg'] if 'away_xg' in match_df.columns else 0
        if away_xg > 0:
            team_stats[home_team]['ratio4_list'].append(row['away_team_goal_count'] / (away_xg + epsilon))
        shots_total = row['away_team_shots_on_target'] + row['away_team_shots_off_target'] + epsilon
        team_stats[home_team]['ratio5_list'].append(row['away_team_goal_count'] / shots_total)
        if row['away_team_possession'] > 0:
            team_stats[home_team]['ratio6_list'].append(
                row['away_team_goal_count'] / (row['away_team_possession'] + epsilon))
        team_stats[home_team]['total_goals_conceded'] += row['away_team_goal_count']
        team_stats[home_team]['num_matches'] += 1

    # 客场比赛统计
    for idx, row in match_df.iterrows():
        away_team = row['away_team_name']
        if away_team not in team_stats:
            print(f"警告: 客队 {away_team} 未找到，跳过比赛 {idx}")
            continue
        team_stats[away_team]['total_goals_conceded'] += row['home_team_goal_count']
        team_stats[away_team]['num_matches'] += 1

    # 汇总数据
    data = []
    for team, stats in team_stats.items():
        num_matches = stats['num_matches']
        average_goals_conceded = stats['total_goals_conceded'] / num_matches if num_matches > 0 else 0
        data.append({
            'team_name': team,
            'ratio1': np.mean(stats['ratio1_list']) if stats['ratio1_list'] else 0,
            'ratio2': np.mean(stats['ratio2_list']) if stats['ratio2_list'] else 0,
            'ratio3': np.mean(stats['ratio3_list']) if stats['ratio3_list'] else 0,
            'ratio4': np.mean(stats['ratio4_list']) if stats['ratio4_list'] else 0,
            'ratio5': np.mean(stats['ratio5_list']) if stats['ratio5_list'] else 0,
            'ratio6': np.mean(stats['ratio6_list']) if stats['ratio6_list'] else 0,
            'total_goals_conceded': stats['total_goals_conceded'],
            'average_goals_conceded': average_goals_conceded
        })

    return pd.DataFrame(data)

# ====================== ELO评分算法实现 ======================
def initialize_elo_scores(team_positions_df):
    """初始化球队ELO分数"""
    teams = team_positions_df['team_name'].unique().tolist()
    team_elo = {team: 1500 for team in teams}
    for team in teams:
        team_data = team_positions_df[team_positions_df['team_name'] == team]
        rank = team_data['points_per_game'].rank().iloc[0]
        team_elo[team] += (20 * (len(teams) - rank))
    return team_elo

def update_elo_scores(elo_scores, home_team, away_team, home_score, away_score, K=30):
    """更新ELO分数"""
    home_elo, away_elo = elo_scores[home_team], elo_scores[away_team]
    expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    expected_away = 1 / (1 + 10 ** ((home_elo - away_elo) / 400))

    if home_score > away_score:
        elo_scores[home_team] += K * (1 - expected_home)
        elo_scores[away_team] += K * (0 - expected_away)
    elif home_score < away_score:
        elo_scores[home_team] += K * (0 - expected_home)
        elo_scores[away_team] += K * (1 - expected_away)
    else:
        elo_scores[home_team] += K * (0.5 - expected_home)
        elo_scores[away_team] += K * (0.5 - expected_away)
    return elo_scores

# ====================== 计算主客场表现差异比值 ======================
def compute_home_away_diff(match_df, team_positions_df, elo_scores):
    """计算每支球队在每个赛季的主客场表现差异比值"""
    epsilon = 1e-8  # 防止除零
    home_away_diff = {}

    for team in team_positions_df['team_name'].unique():
        for season in match_df['season'].unique():
            team_matches = match_df[
                ((match_df['home_team_name'] == team) | (match_df['away_team_name'] == team)) &
                (match_df['season'] == season)
            ]
            if team_matches.empty:
                continue

            opponents = set(team_matches['home_team_name'].unique()) | set(team_matches['away_team_name'].unique())
            opponents.remove(team)

            diff_ratios = []
            weights = []

            for opponent in opponents:
                home_match = team_matches[
                    (team_matches['home_team_name'] == team) & (team_matches['away_team_name'] == opponent)]
                away_match = team_matches[
                    (team_matches['away_team_name'] == team) & (team_matches['home_team_name'] == opponent)]

                if not home_match.empty and not away_match.empty:
                    home_xg = home_match['home_xg'].values[0] if 'home_xg' in home_match.columns else 0
                    away_xg = away_match['away_xg'].values[0] if 'away_xg' in away_match.columns else 0
                    home_conceded_xg = home_match['away_xg'].values[0] if 'away_xg' in home_match.columns else 0
                    away_conceded_xg = away_match['home_xg'].values[0] if 'home_xg' in away_match.columns else 0

                    ratio_xg = home_xg / (away_xg + epsilon)
                    ratio_conceded = home_conceded_xg / (away_conceded_xg + epsilon)
                    diff_ratio = (ratio_xg + ratio_conceded) / 2

                    opponent_elo = elo_scores.get(opponent, 1500)
                    diff_ratios.append(diff_ratio)
                    weights.append(opponent_elo)

            if diff_ratios:
                weighted_diff = np.average(diff_ratios, weights=weights)
                home_away_diff[f"{team}_{season}"] = weighted_diff
            else:
                home_away_diff[f"{team}_{season}"] = 1.0

    return home_away_diff