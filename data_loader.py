# data_loader.py
# 描述: 负责加载、合并和预处理来自所有联赛和赛季的数据。

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from feature_engineering import compute_defensive_stats, initialize_elo_scores, compute_home_away_diff


def load_all_league_data(base_path, leagues, seasons):
    """加载所有联赛数据并进行预处理，同时计算主客场差异"""
    all_team_positions, all_match_positions = [], []
    all_raw_matches = []  # 用于计算主客场差异
    elo_scores = None  # 将在第一次加载后初始化

    # 第一遍：加载数据并准备计算
    for country_name, league_name in leagues:
        for season in seasons:
            print(f"加载数据: {country_name} - {league_name} - {season}")
            team_file = os.path.join(base_path, f"{country_name}-{league_name}-teams-{season}-stats.csv")
            match_file = os.path.join(base_path, f"{country_name}-{league_name}-matches-{season}-stats.csv")
            if not os.path.exists(team_file) or not os.path.exists(match_file):
                print(f"警告: {country_name} - {league_name} - {season} 文件缺失")
                continue

            team_df = pd.read_csv(team_file)
            match_df = pd.read_csv(match_file)

            team_df['team_name'] = team_df.get('common_name', team_df.get('team_name', None)).str.strip().str.lower()
            match_df['home_team_name'] = match_df['home_team_name'].str.strip().str.lower()
            match_df['away_team_name'] = match_df['away_team_name'].str.strip().str.lower()
            match_df['season'] = season  # 添加赛季标识

            team_names = team_df['team_name'].unique()
            match_df = match_df[
                match_df['home_team_name'].isin(team_names) & match_df['away_team_name'].isin(team_names)]

            if match_df.empty: continue

            # 初始化ELO
            if elo_scores is None:
                elo_scores = initialize_elo_scores(team_df)

            defensive_stats_df = compute_defensive_stats(match_df, team_df)
            if defensive_stats_df.empty: continue

            team_df = team_df.merge(defensive_stats_df, on='team_name', how='left')
            for col in ['ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5', 'ratio6', 'total_goals_conceded',
                        'average_goals_conceded']:
                if col not in team_df.columns: team_df[col] = 0

            # 归一化防守分
            max_conceded, min_conceded = team_df['total_goals_conceded'].max(), team_df['total_goals_conceded'].min()
            if max_conceded > min_conceded:
                team_df['normalized_defense_score'] = (max_conceded - team_df['total_goals_conceded']) / (
                            max_conceded - min_conceded + 1e-8)
            else:
                team_df['normalized_defense_score'] = 0

            # PCA 降维
            defensive_cols = [c for c in
                              ['goals_conceded', 'total_goals_conceded', 'ratio1', 'ratio2', 'ratio3', 'ratio4',
                               'ratio5', 'ratio6', 'normalized_defense_score'] if c in team_df.columns]
            team_df_defensive = team_df[defensive_cols].fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(team_df_defensive)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            team_positions = pd.DataFrame({
                'team_name': team_df['team_name'], 'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1],
                'points_per_game': team_df['points_per_game'], 'league': league_name, 'season': season,
                'ratio1': team_df['ratio1'], 'ratio2': team_df['ratio2'], 'ratio3': team_df['ratio3'],
                'ratio4': team_df['ratio4'], 'ratio5': team_df['ratio5'], 'ratio6': team_df['ratio6'],
                'normalized_defense_score': team_df['normalized_defense_score'],
                'total_goals_conceded': team_df['total_goals_conceded'],
                'average_goals_conceded': team_df['average_goals_conceded']
            })
            team_positions['team_season'] = league_name + '_' + team_positions['team_name'] + '_' + team_positions[
                'season']
            all_team_positions.append(team_positions)

            # 比赛数据PCA
            match_df_numeric = match_df.select_dtypes(include=np.number).fillna(0)
            scaler_match = StandardScaler()
            X_match_scaled = scaler_match.fit_transform(match_df_numeric)
            pca_match = PCA(n_components=2)
            X_pca_match = pca_match.fit_transform(X_match_scaled)

            match_positions = pd.DataFrame({
                'home_team_name': match_df['home_team_name'], 'away_team_name': match_df['away_team_name'],
                'PC1': X_pca_match[:, 0], 'PC2': X_pca_match[:, 1], 'league': league_name, 'season': season
            })
            all_match_positions.append(match_positions)
            all_raw_matches.append(match_df)

    all_team_positions_df = pd.concat(all_team_positions, ignore_index=True)
    all_match_positions_df = pd.concat(all_match_positions, ignore_index=True)
    all_raw_matches_df = pd.concat(all_raw_matches, ignore_index=True)

    # 计算主客场差异比值
    home_away_diff = compute_home_away_diff(all_raw_matches_df, all_team_positions_df, elo_scores)
    all_team_positions_df['home_away_diff'] = all_team_positions_df.apply(
        lambda row: home_away_diff.get(f"{row['team_name']}_{row['season']}", 1.0), axis=1
    )

    return all_team_positions_df, all_match_positions_df