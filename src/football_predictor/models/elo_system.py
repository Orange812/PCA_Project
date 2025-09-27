#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
足球博彩预测分析系统
基于ELO评分和机器学习的进球数预测系统
"""

import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FootballBettingSystem:
    """足球博彩预测分析系统主类"""
    
    def __init__(self, data_path: str = "data", db_path: str = "football_betting.db"):
        self.data_path = data_path
        self.db_path = db_path
        self.home_advantage = 0.1  # 主客场优势系数，将通过数据计算得出
        self.elo_k_factor = 30
        self.initial_elo = 1500
        
        # 初始化数据库
        self._init_database()
        
    def _init_database(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建球队表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT UNIQUE,
                league TEXT,
                country TEXT,
                current_elo REAL DEFAULT 1500,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建ELO历史记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS elo_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT,
                season TEXT,
                match_date TEXT,
                elo_before REAL,
                elo_after REAL,
                opponent TEXT,
                home_away TEXT,
                result TEXT,
                goals_for INTEGER,
                goals_against INTEGER,
                FOREIGN KEY (team_name) REFERENCES teams (team_name)
            )
        ''')
        
        # 创建比赛数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season TEXT,
                league TEXT,
                match_date TEXT,
                home_team TEXT,
                away_team TEXT,
                home_goals INTEGER,
                away_goals INTEGER,
                home_elo_before REAL,
                away_elo_before REAL,
                home_elo_after REAL,
                away_elo_after REAL,
                is_valid BOOLEAN DEFAULT 1,
                FOREIGN KEY (home_team) REFERENCES teams (team_name),
                FOREIGN KEY (away_team) REFERENCES teams (team_name)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def calculate_home_advantage(self) -> float:
        """计算主客场优势系数"""
        print("正在计算主客场优势系数...")
        
        all_matches = []
        leagues = [
            ("england", "premier-league"), ("germany", "bundesliga"), ("spain", "la-liga"),
            ("france", "ligue-1"), ("italy", "serie-a"), ("netherlands", "eredivisie")
        ]
        seasons = ["2020-to-2021", "2021-to-2022", "2022-to-2023", "2023-to-2024"]
        
        for country, league in leagues:
            for season in seasons:
                match_file = os.path.join(self.data_path, f"{country}-{league}-matches-{season}-stats.csv")
                if os.path.exists(match_file):
                    try:
                        df = pd.read_csv(match_file)
                        # 数据清洗：剔除净胜球大于4的比赛
                        df = df[abs(df['home_team_goal_count'] - df['away_team_goal_count']) <= 4]
                        all_matches.append(df[['home_team_goal_count', 'away_team_goal_count']])
                    except Exception as e:
                        print(f"读取文件失败 {match_file}: {e}")
        
        if not all_matches:
            print("警告: 没有找到有效的比赛数据，使用默认主客场优势系数 0.1")
            return 0.1
            
        # 合并所有比赛数据
        combined_matches = pd.concat(all_matches, ignore_index=True)
        
        # 计算主客场胜率
        home_wins = len(combined_matches[combined_matches['home_team_goal_count'] > combined_matches['away_team_goal_count']])
        away_wins = len(combined_matches[combined_matches['home_team_goal_count'] < combined_matches['away_team_goal_count']])
        total_decisive = home_wins + away_wins
        
        if total_decisive == 0:
            return 0.1
            
        home_win_rate = home_wins / total_decisive
        away_win_rate = away_wins / total_decisive
        
        # 主客场优势系数 = (主队胜率 - 客队胜率) / 2
        home_advantage = (home_win_rate - away_win_rate) / 2
        
        print(f"主队胜率: {home_win_rate:.3f}")
        print(f"客队胜率: {away_win_rate:.3f}")
        print(f"计算得出的主客场优势系数: {home_advantage:.3f}")
        
        return max(0.05, min(0.2, home_advantage))  # 限制在合理范围内
        
    def initialize_team_elo(self, team_positions_df: pd.DataFrame, league: str, season: str) -> Dict[str, float]:
        """初始化球队ELO分数"""
        teams = team_positions_df['team_name'].unique().tolist()
        team_elo = {}
        
        for team in teams:
            team_data = team_positions_df[team_positions_df['team_name'] == team]
            if not team_data.empty:
                # 基于积分排名初始化ELO
                ppg = team_data['points_per_game'].iloc[0] if 'points_per_game' in team_data.columns else 1.5
                rank_bonus = (ppg - 1.5) * 100  # 基于场均积分的调整
                team_elo[team] = self.initial_elo + rank_bonus
            else:
                team_elo[team] = self.initial_elo
                
        return team_elo
    
    def update_elo_with_home_advantage(self, home_elo: float, away_elo: float, 
                                     home_goals: int, away_goals: int) -> Tuple[float, float]:
        """更新ELO分数（考虑主客场优势）"""
        # 计算期望胜率（加入主客场优势）
        home_advantage_points = self.home_advantage * 400
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo - home_advantage_points) / 400))
        expected_away = 1 - expected_home
        
        # 确定比赛结果
        if home_goals > away_goals:
            home_result, away_result = 1, 0
        elif home_goals < away_goals:
            home_result, away_result = 0, 1
        else:
            home_result, away_result = 0.5, 0.5
        
        # 更新ELO分数
        new_home_elo = home_elo + self.elo_k_factor * (home_result - expected_home)
        new_away_elo = away_elo + self.elo_k_factor * (away_result - expected_away)
        
        return new_home_elo, new_away_elo
    
    def process_league_season(self, country: str, league: str, season: str):
        """处理单个联赛赛季的数据"""
        print(f"处理数据: {country} - {league} - {season}")
        
        team_file = os.path.join(self.data_path, f"{country}-{league}-teams-{season}-stats.csv")
        match_file = os.path.join(self.data_path, f"{country}-{league}-matches-{season}-stats.csv")
        
        if not os.path.exists(team_file) or not os.path.exists(match_file):
            print(f"警告: 文件缺失 {country}-{league}-{season}")
            return None
        
        try:
            team_df = pd.read_csv(team_file)
            match_df = pd.read_csv(match_file)
            
            # 统一球队名称格式
            team_df['team_name'] = team_df.get('common_name', team_df.get('team_name', '')).str.strip().str.lower()
            match_df['home_team_name'] = match_df['home_team_name'].str.strip().str.lower()
            match_df['away_team_name'] = match_df['away_team_name'].str.strip().str.lower()
            
            # 数据清洗：剔除净胜球大于4的比赛
            original_count = len(match_df)
            match_df = match_df[abs(match_df['home_team_goal_count'] - match_df['away_team_goal_count']) <= 4]
            filtered_count = len(match_df)
            
            if filtered_count < original_count:
                print(f"数据清洗: 剔除了 {original_count - filtered_count} 场净胜球>4的比赛")
            
            # 过滤有效比赛
            team_names = set(team_df['team_name'].unique())
            match_df = match_df[
                match_df['home_team_name'].isin(team_names) & 
                match_df['away_team_name'].isin(team_names)
            ]
            
            if match_df.empty:
                print("警告: 无有效比赛数据")
                return None
            
            # 初始化ELO分数
            team_elo = self.initialize_team_elo(team_df, league, season)
            
            # 存储球队信息到数据库
            self._store_teams(team_elo, league, country)
            
            # 按日期排序比赛
            if 'date_GMT' in match_df.columns:
                match_df = match_df.sort_values('date_GMT')
            
            # 处理每场比赛并更新ELO
            match_results = []
            odds_cols = [
                'odds_ft_home_team_win', 'odds_ft_draw', 'odds_ft_away_team_win',
                'odds_ft_over15', 'odds_ft_over25', 'odds_ft_over35',
                'odds_btts_yes', 'odds_btts_no'
            ]
            for _, row in match_df.iterrows():
                home_team = row['home_team_name']
                away_team = row['away_team_name']
                home_goals = row['home_team_goal_count']
                away_goals = row['away_team_goal_count']
                match_date = row.get('date_GMT', '')

                # 获取赔率数据, 如果CSV中不存在该列则默认为0.0
                odds_values = {col: row.get(col, 0.0) for col in odds_cols}
                
                # 获取比赛前的ELO分数
                home_elo_before = team_elo[home_team]
                away_elo_before = team_elo[away_team]
                
                # 更新ELO分数
                home_elo_after, away_elo_after = self.update_elo_with_home_advantage(
                    home_elo_before, away_elo_before, home_goals, away_goals
                )
                
                # 更新字典中的ELO分数
                team_elo[home_team] = home_elo_after
                team_elo[away_team] = away_elo_after
                
                # 记录比赛结果
                match_data = {
                    'season': season,
                    'league': f"{country}-{league}",
                    'match_date': match_date,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'home_elo_before': home_elo_before,
                    'away_elo_before': away_elo_before,
                    'home_elo_after': home_elo_after,
                    'away_elo_after': away_elo_after
                }
                match_data.update(odds_values)
                match_results.append(match_data)
            
            # 存储比赛数据到数据库
            self._store_matches(match_results)
            
            return team_elo
            
        except Exception as e:
            print(f"处理数据时出错 {country}-{league}-{season}: {e}")
            return None
    
    def _store_teams(self, team_elo: Dict[str, float], league: str, country: str):
        """存储球队信息到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for team_name, elo in team_elo.items():
            cursor.execute('''
                INSERT OR REPLACE INTO teams (team_name, league, country, current_elo)
                VALUES (?, ?, ?, ?)
            ''', (team_name, league, country, elo))
        
        conn.commit()
        conn.close()
    
    def _store_matches(self, match_results: List[Dict]):
        """存储比赛数据到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 定义基础列和赔率列
        base_cols = [
            'season', 'league', 'match_date', 'home_team', 'away_team', 
            'home_goals', 'away_goals', 'home_elo_before', 'away_elo_before', 
            'home_elo_after', 'away_elo_after'
        ]
        odds_cols = [
            'odds_ft_home_team_win', 'odds_ft_draw', 'odds_ft_away_team_win',
            'odds_ft_over15', 'odds_ft_over25', 'odds_ft_over35',
            'odds_btts_yes', 'odds_btts_no'
        ]
        
        # 动态构建SQL语句
        all_columns = base_cols + odds_cols
        columns_str = ', '.join(all_columns)
        placeholders_str = ', '.join(['?'] * len(all_columns))
        sql = f"INSERT INTO matches ({columns_str}) VALUES ({placeholders_str})"

        for match in match_results:
            # 确保所有列都存在，为缺失的赔率列提供默认值0.0
            values = tuple(match.get(col, 0.0) for col in all_columns)
            try:
                cursor.execute(sql, values)
            except sqlite3.Error as e:
                print(f"数据库插入失败: {e}")
                print(f"失败的值: {values}")

        conn.commit()
        conn.close()
    
    def calculate_elo_weighted_goals(self, team_name: str, recent_matches: int = 10) -> Dict[str, float]:
        """计算ELO加权的进球数预测"""
        conn = sqlite3.connect(self.db_path)
        
        # 获取球队最近的比赛记录
        query = '''
            SELECT m.*, 
                   CASE WHEN m.home_team = ? THEN m.away_team ELSE m.home_team END as opponent,
                   CASE WHEN m.home_team = ? THEN 'home' ELSE 'away' END as venue,
                   CASE WHEN m.home_team = ? THEN m.home_goals ELSE m.away_goals END as goals_for,
                   CASE WHEN m.home_team = ? THEN m.away_goals ELSE m.home_goals END as goals_against,
                   CASE WHEN m.home_team = ? THEN m.home_elo_before ELSE m.away_elo_before END as team_elo,
                   CASE WHEN m.home_team = ? THEN m.away_elo_before ELSE m.home_elo_before END as opponent_elo
            FROM matches m
            WHERE (m.home_team = ? OR m.away_team = ?)
            ORDER BY m.match_date DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=[team_name] * 8 + [recent_matches])
        conn.close()
        
        if df.empty:
            return {'predicted_goals': 1.5, 'confidence': 0.0}
        
        # 计算ELO加权的进球数
        total_weighted_goals = 0
        total_weights = 0
        
        current_elo = df.iloc[0]['team_elo']  # 使用最近一场比赛的ELO作为当前ELO
        
        for i, row in df.iterrows():
            goals = row['goals_for']
            opponent_elo = row['opponent_elo']
            team_elo = row['team_elo']
            
            # 时间衰减权重（最近的比赛权重更高）
            time_weight = 0.9 ** i
            
            # ELO差异权重（对手实力越接近当前对手，权重越高）
            elo_diff = abs(opponent_elo - current_elo)
            elo_weight = np.exp(-elo_diff / 200)  # 200是调节参数
            
            # 综合权重
            combined_weight = time_weight * elo_weight
            
            total_weighted_goals += goals * combined_weight
            total_weights += combined_weight
        
        if total_weights == 0:
            predicted_goals = df['goals_for'].mean()
            confidence = 0.5
        else:
            predicted_goals = total_weighted_goals / total_weights
            confidence = min(1.0, total_weights / recent_matches)
        
        return {
            'predicted_goals': predicted_goals,
            'confidence': confidence,
            'recent_matches_count': len(df)
        }
    
    def predict_match_goals(self, home_team: str, away_team: str, recent_matches: int = 10) -> Dict:
        """预测比赛的进球数"""
        # 获取两队的ELO加权进球预测
        home_prediction = self.calculate_elo_weighted_goals(home_team, recent_matches)
        away_prediction = self.calculate_elo_weighted_goals(away_team, recent_matches)
        
        # 获取当前ELO分数
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT current_elo FROM teams WHERE team_name = ?', (home_team,))
        home_elo_result = cursor.fetchone()
        home_elo = home_elo_result[0] if home_elo_result else self.initial_elo
        
        cursor.execute('SELECT current_elo FROM teams WHERE team_name = ?', (away_team,))
        away_elo_result = cursor.fetchone()
        away_elo = away_elo_result[0] if away_elo_result else self.initial_elo
        
        conn.close()
        
        # 基于ELO差异调整预测
        elo_diff = home_elo - away_elo
        elo_adjustment = elo_diff / 400 * 0.3  # 调节系数
        
        # 主客场优势调整
        home_advantage_goals = self.home_advantage * 0.5  # 转换为进球数优势
        
        # 最终预测
        home_goals_predicted = max(0.1, home_prediction['predicted_goals'] + elo_adjustment + home_advantage_goals)
        away_goals_predicted = max(0.1, away_prediction['predicted_goals'] - elo_adjustment)
        
        total_goals_predicted = home_goals_predicted + away_goals_predicted
        
        # 计算置信度
        avg_confidence = (home_prediction['confidence'] + away_prediction['confidence']) / 2
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_goals_predicted': round(home_goals_predicted, 2),
            'away_goals_predicted': round(away_goals_predicted, 2),
            'total_goals_predicted': round(total_goals_predicted, 2),
            'home_elo': home_elo,
            'away_elo': away_elo,
            'confidence': round(avg_confidence, 2),
            'home_win_probability': self._calculate_win_probability(home_elo, away_elo, 'home'),
            'draw_probability': self._calculate_win_probability(home_elo, away_elo, 'draw'),
            'away_win_probability': self._calculate_win_probability(home_elo, away_elo, 'away')
        }
    
    def _calculate_win_probability(self, home_elo: float, away_elo: float, outcome: str) -> float:
        """计算胜平负概率"""
        home_advantage_points = self.home_advantage * 400
        
        if outcome == 'home':
            expected = 1 / (1 + 10 ** ((away_elo - home_elo - home_advantage_points) / 400))
            return round(expected * 0.85, 3)  # 调整系数，因为平局概率需要考虑
        elif outcome == 'away':
            expected = 1 / (1 + 10 ** ((home_elo - away_elo + home_advantage_points) / 400))
            return round(expected * 0.85, 3)
        else:  # draw
            return round(0.3, 3)  # 简化的平局概率
    
    def find_optimal_recent_matches(self, test_matches: int = 100) -> int:
        """通过交叉验证找到最优的历史比赛场次数"""
        print("正在寻找最优的历史比赛场次数...")
        
        conn = sqlite3.connect(self.db_path)
        
        # 获取测试数据
        query = '''
            SELECT home_team, away_team, home_goals, away_goals
            FROM matches
            ORDER BY RANDOM()
            LIMIT ?
        '''
        test_df = pd.read_sql_query(query, conn, params=[test_matches])
        conn.close()
        
        if test_df.empty:
            print("警告: 没有足够的测试数据，使用默认值 10")
            return 10
        
        best_x = 10
        best_accuracy = 0
        
        for x in range(5, 21):  # 测试5-20场的范围
            correct_predictions = 0
            total_predictions = 0
            
            for _, row in test_df.iterrows():
                try:
                    prediction = self.predict_match_goals(row['home_team'], row['away_team'], x)
                    
                    # 简单的准确性评估：预测进球数与实际进球数的差异
                    home_diff = abs(prediction['home_goals_predicted'] - row['home_goals'])
                    away_diff = abs(prediction['away_goals_predicted'] - row['away_goals'])
                    
                    # 如果预测误差在1个球以内，认为是正确的
                    if home_diff <= 1 and away_diff <= 1:
                        correct_predictions += 1
                    
                    total_predictions += 1
                    
                except Exception:
                    continue
            
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_x = x
        
        print(f"最优历史比赛场次数: {best_x}, 准确率: {best_accuracy:.3f}")
        return best_x
    
    def run_full_analysis(self):
        """运行完整的分析流程"""
        print("=== 足球博彩预测分析系统 ===")
        print("开始数据处理...")
        
        # 1. 计算主客场优势系数
        self.home_advantage = self.calculate_home_advantage()
        print(f"使用主客场优势系数: {self.home_advantage:.3f}")
        
        # 2. 处理所有联赛数据
        leagues = [
            ("england", "premier-league"), ("germany", "bundesliga"), ("spain", "la-liga"),
            ("france", "ligue-1"), ("italy", "serie-a"), ("netherlands", "eredivisie"),
            ("portugal", "liga-nos"), ("denmark", "superliga"), ("england", "championship"),
            ("italy", "serie-b"), ("germany", "2-bundesliga")
        ]
        
        seasons = [
            "2020-to-2021", "2021-to-2022", "2022-to-2023", "2023-to-2024"
        ]
        
        processed_count = 0
        for country, league in leagues:
            for season in seasons:
                result = self.process_league_season(country, league, season)
                if result is not None:
                    processed_count += 1
        
        print(f"成功处理了 {processed_count} 个联赛赛季的数据")
        
        # 3. 寻找最优历史比赛场次数
        optimal_x = self.find_optimal_recent_matches()
        
        print(f"\n=== 系统初始化完成 ===")
        print(f"主客场优势系数: {self.home_advantage:.3f}")
        print(f"最优历史比赛场次数: {optimal_x}")
        print(f"数据库路径: {self.db_path}")
        
        return optimal_x

# 使用示例
if __name__ == "__main__":
    # 创建系统实例
    system = FootballBettingSystem()
    
    # 运行完整分析
    optimal_recent_matches = system.run_full_analysis()
    
    # 示例预测
    print("\n=== 示例预测 ===")
    try:
        prediction = system.predict_match_goals("manchester city", "arsenal", optimal_recent_matches)
        print(f"比赛预测: {prediction['home_team']} vs {prediction['away_team']}")
        print(f"预测进球: {prediction['home_goals_predicted']} - {prediction['away_goals_predicted']}")
        print(f"总进球数: {prediction['total_goals_predicted']}")
        print(f"胜负概率: 主胜 {prediction['home_win_probability']}, 平局 {prediction['draw_probability']}, 客胜 {prediction['away_win_probability']}")
        print(f"置信度: {prediction['confidence']}")
    except Exception as e:
        print(f"预测示例失败: {e}")