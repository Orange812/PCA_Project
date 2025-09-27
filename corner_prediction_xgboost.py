# -*- coding: utf-8 -*-
"""
@File: corner_prediction_xgboost.py
@Description:
学习曲线分析版（已修复）。
本脚本用于通过在不同大小的数据集上训练模型，来分析增加数据量是否能提升模型性能。
@Author: Gemini
@Date: 2025-08-23
"""

import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import re
import json

DATA_DIR = 'data'

def load_and_process_team_stats(data_path: str) -> pd.DataFrame:
    """加载、处理并整合所有球队的赛季统计数据，区分主客场。"""
    team_files = glob.glob(os.path.join(data_path, '*-teams-*-stats.csv'))
    team_dfs = []
    for file in team_files:
        df = pd.read_csv(file)
        # Handle potential division by zero if a team has no home/away matches
        df['avg_shots_home'] = df['shots_home'] / (df['matches_played_home'] + 1e-6)
        df['avg_shots_away'] = df['shots_away'] / (df['matches_played_away'] + 1e-6)
        
        # Directly use or rename provided columns
        df['avg_possession_home'] = df['average_possession_home']
        df['avg_possession_away'] = df['average_possession_away']
        df['avg_corners_home'] = df['corners_per_match_home']
        df['avg_corners_away'] = df['corners_per_match_away']
        df['avg_xg_home'] = df['xg_for_avg_home']
        df['avg_xg_away'] = df['xg_for_avg_away']
        
        processed_df = df[[
            'team_name', 'season',
            'avg_possession_home', 'avg_shots_home', 'avg_corners_home', 'avg_xg_home',
            'avg_possession_away', 'avg_shots_away', 'avg_corners_away', 'avg_xg_away'
        ]]
        team_dfs.append(processed_df)
    all_teams_stats = pd.concat(team_dfs, ignore_index=True)
    all_teams_stats.drop_duplicates(subset=['team_name', 'season'], keep='last', inplace=True)
    all_teams_stats.set_index(['team_name', 'season'], inplace=True)
    return all_teams_stats

def create_feature_dataset(team_stats_df: pd.DataFrame, data_path: str, name_map_file: str) -> (pd.DataFrame, pd.Series):
    """使用名称映射文件和赔率数据，并根据主客场情景创建增强版特征集。"""
    with open(name_map_file, 'r', encoding='utf-8') as f:
        team_name_map = json.load(f)

    match_files = glob.glob(os.path.join(data_path, '*-matches-*-stats.csv'))
    all_matches_dfs = []
    season_pattern = re.compile(r'(\d{4}-to-\d{4})')
    for file in match_files:
        match = season_pattern.search(os.path.basename(file))
        if not match:
            continue
        season = match.group(1).replace('-to-', '/')
        df = pd.read_csv(file)
        df['season'] = season
        all_matches_dfs.append(df)
    matches_df = pd.concat(all_matches_dfs, ignore_index=True)

    matches_df['home_team_name_mapped'] = matches_df['home_team_name'].map(team_name_map).fillna(matches_df['home_team_name'])
    matches_df['away_team_name_mapped'] = matches_df['away_team_name'].map(team_name_map).fillna(matches_df['away_team_name'])

    odds_cols = ['odds_ft_home_team_win', 'odds_ft_draw', 'odds_ft_away_team_win', 'odds_ft_over25', 'odds_btts_yes']
    matches_df[odds_cols] = matches_df[odds_cols].replace(0, np.nan)
    matches_df.dropna(subset=odds_cols, inplace=True)

    prob_h = 1 / matches_df['odds_ft_home_team_win']
    prob_d = 1 / matches_df['odds_ft_draw']
    prob_a = 1 / matches_df['odds_ft_away_team_win']
    overround = prob_h + prob_d + prob_a
    matches_df['home_win_fair_prob'] = prob_h / overround
    matches_df['strength_disparity'] = (prob_h / overround) - (prob_a / overround)
    matches_df['over_25_prob'] = 1 / matches_df['odds_ft_over25']
    matches_df['btts_yes_prob'] = 1 / matches_df['odds_btts_yes']

    # Merge for home team stats
    merged_df = pd.merge(matches_df, team_stats_df, left_on=['home_team_name_mapped', 'season'], right_index=True, how='inner')
    
    # Merge for away team stats, using suffixes to distinguish columns
    merged_df = pd.merge(merged_df, team_stats_df, left_on=['away_team_name_mapped', 'season'], right_index=True, how='inner', suffixes=('_home_team', '_away_team'))

    # Create the final feature set by picking the correct contextual stats
    # Home team features are from its HOME stats
    merged_df['home_avg_possession'] = merged_df['avg_possession_home_home_team']
    merged_df['home_avg_shots'] = merged_df['avg_shots_home_home_team']
    merged_df['home_avg_corners'] = merged_df['avg_corners_home_home_team']
    merged_df['home_avg_xg'] = merged_df['avg_xg_home_home_team']
    
    # Away team features are from its AWAY stats
    merged_df['away_avg_possession'] = merged_df['avg_possession_away_away_team']
    merged_df['away_avg_shots'] = merged_df['avg_shots_away_away_team']
    merged_df['away_avg_corners'] = merged_df['avg_corners_away_away_team']
    merged_df['away_avg_xg'] = merged_df['avg_xg_away_away_team']

    feature_columns = [
        'home_avg_possession', 'home_avg_shots', 'home_avg_corners', 'home_avg_xg',
        'away_avg_possession', 'away_avg_shots', 'away_avg_corners', 'away_avg_xg',
        'home_win_fair_prob', 'strength_disparity', 'over_25_prob', 'btts_yes_prob'
    ]
    
    merged_df['total_corners'] = merged_df['home_team_corner_count'] + merged_df['away_team_corner_count']
    # 数据清洗：确保目标变量为非负数，以满足泊松回归的要求
    merged_df = merged_df[merged_df['total_corners'] >= 0]
    merged_df.dropna(subset=feature_columns + ['total_corners'], inplace=True)
    
    X = merged_df[feature_columns]
    y = merged_df['total_corners']
    
    print(f"成功创建情景化特征数据集，包含 {len(X)} 场比赛样本。")
    return X, y

def main():
    """主函数，执行学习曲线分析实验。"""
    print("--- 开始执行学习曲线分析任务 ---")
    try:
        team_stats_df = load_and_process_team_stats(DATA_DIR)
        X, y = create_feature_dataset(team_stats_df, DATA_DIR, 'team_name_map.json')
        if X is None or y is None or X.empty:
            return
    except (FileNotFoundError, ValueError) as e:
        print(f"在数据处理阶段发生错误: {e}")
        return

    # 1. 划分一个全局的训练集和固定的测试集
    X_train_master, X_test, y_train_master, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"数据准备完成：全局训练集 {len(X_train_master)} 条，固定测试集 {len(X_test)} 条。")

    # 2. 定义训练规模
    training_sizes_pct = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []

    print("\n--- 开始迭代训练与评估 ---")
    # 设置随机数种子以便复现
    np.random.seed(42)
    for pct in training_sizes_pct:
        # 3. 从全局训练集中抽取子集 (已修复)
        subset_size = int(len(X_train_master) * pct)
        if pct < 1.0:
            train_indices = X_train_master.index
            subset_indices = np.random.choice(train_indices, size=subset_size, replace=False)
            X_train_subset = X_train_master.loc[subset_indices]
            y_train_subset = y_train_master.loc[subset_indices]
        else:
            X_train_subset, y_train_subset = X_train_master, y_train_master
        
        print(f"正在使用 {subset_size} ({pct*100:.0f}%) 个训练样本进行训练...")

        # 4. 训练模型
        model = xgb.XGBRegressor(
            objective='count:poisson', n_estimators=500, learning_rate=0.05,
            max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        model.fit(X_train_subset, y_train_subset, verbose=False)

        # 5. 在固定的测试集上评估
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        results.append({'train_size': subset_size, 'mae': mae})

    # 6. 报告结果
    print("\n--- 学习曲线分析结果 ---")
    print("训练样本数   |   在固定测试集上的MAE")
    print("------------------------------------------")
    for res in results:
        print(f"{res['train_size']:<14} |   {res['mae']:.4f}")

    # 7. 解读结果
    print("\n--- 结果解读 ---")
    initial_mae = results[0]['mae']
    final_mae = results[-1]['mae']
    improvement = initial_mae - final_mae

    if improvement > 0.05: # 如果总提升超过一个比较显著的阈值
        print("结论：模型的性能随着数据量的增加而稳定提升。")
        print(f"从最小数据集到最大数据集，MAE降低了 {improvement:.4f}。")
        print("这清晰地表明，模型还没有‘学透’，它依然可以从更多的数据中获益。")
        print("\n建议：增加更多赛季或更多联赛的数据，有望进一步提升模型的预测精度。")
    else:
        print("结论：模型的性能在训练数据量达到一定规模后，趋于平稳。")
        print(f"从最小数据集到最大数据集，MAE仅降低了 {improvement:.4f}。")
        print("这表明对于现有特征，模型的能力已趋于饱和。")
        print("\n建议：简单地增加更多同类型数据可能效果有限。未来的优化方向应重点放在‘特征工程’上，例如引入‘近期状态’等更能反映比赛动态的新特征。")

    print("\n--- 任务执行完毕 ---")

if __name__ == '__main__':
    main()