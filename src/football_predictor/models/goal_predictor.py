#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级进球预测模块
基于ELO加权和机器学习的进球数预测
"""

import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class AdvancedGoalPredictor:
    """高级进球预测器"""
    
    def __init__(self, db_path: str = "football_betting.db"):
        self.db_path = db_path
        self.home_model = None
        self.away_model = None
        self.is_trained = False
        self.odds_cols = [
            'odds_ft_home_team_win', 'odds_ft_draw', 'odds_ft_away_team_win',
            'odds_ft_over15', 'odds_ft_over25', 'odds_ft_over35',
            'odds_btts_yes', 'odds_btts_no'
        ]
        
    def _calculate_odds_features(self, match_odds: Dict[str, float]) -> Dict[str, float]:
        """根据赔率计算特征"""
        odds_features = {}

        # A组特征：市场对“比赛格局”的判断
        h, d, a = match_odds.get('odds_ft_home_team_win', 0), match_odds.get('odds_ft_draw', 0), match_odds.get('odds_ft_away_team_win', 0)
        if h > 0 and d > 0 and a > 0:
            p_h, p_d, p_a = 1/h, 1/d, 1/a
            overround = p_h + p_d + p_a
            odds_features['market_prob_h'] = p_h / overround
            odds_features['market_prob_d'] = p_d / overround
            odds_features['market_prob_a'] = p_a / overround
        else:
            odds_features['market_prob_h'] = -1.0
            odds_features['market_prob_d'] = -1.0
            odds_features['market_prob_a'] = -1.0

        btts_y, btts_n = match_odds.get('odds_btts_yes', 0), match_odds.get('odds_btts_no', 0)
        if btts_y > 0 and btts_n > 0:
            p_y, p_n = 1/btts_y, 1/btts_n
            overround = p_y + p_n
            odds_features['market_prob_btts_yes'] = p_y / overround
        else:
            odds_features['market_prob_btts_yes'] = -1.0

        # B组特征：市场对“总进球数”的直接判断
        o15, o25, o35 = match_odds.get('odds_ft_over15', 0), match_odds.get('odds_ft_over25', 0), match_odds.get('odds_ft_over35', 0)
        odds_features['market_prob_o15'] = 1/o15 if o15 > 0 else -1.0
        odds_features['market_prob_o25'] = 1/o25 if o25 > 0 else -1.0
        odds_features['market_prob_o35'] = 1/o35 if o35 > 0 else -1.0

        # 复合特征：市场隐含总进球数
        if o25 > 0:
            # 假设Under赔率与Over赔率有相似的抽水
            p_o25 = 1 / o25
            # 简化估算，假设抽水为5%，则公平概率约为 p_o25 / 1.05
            fair_p_o25 = p_o25 / 1.05 
            fair_p_u25 = 1 - fair_p_o25
            # 一个简单的估算公式
            implied_goals = 2.5 + (fair_p_o25 - fair_p_u25) * 2 
            odds_features['market_implied_total_goals'] = implied_goals
        else:
            odds_features['market_implied_total_goals'] = -1.0
            
        return odds_features

    def extract_features(self, team_name: str, opponent_name: str, is_home: bool, 
                         match_odds: Dict[str, float], recent_matches: int = 15) -> Dict[str, float]:
        """提取球队特征用于机器学习模型（已集成赔率特征）"""
        conn = sqlite3.connect(self.db_path)
        
        # 获取球队最近比赛数据 (这部分逻辑不变)
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
        
        opponent_query = '''
            SELECT AVG(CASE WHEN home_team = ? THEN away_goals ELSE home_goals END) as avg_goals_conceded,
                   AVG(CASE WHEN home_team = ? THEN home_goals ELSE away_goals END) as avg_goals_scored
            FROM matches
            WHERE (home_team = ? OR away_team = ?)
            ORDER BY match_date DESC
            LIMIT ?
        '''
        opponent_df = pd.read_sql_query(opponent_query, conn, params=[opponent_name] * 4 + [recent_matches])
        conn.close()
        
        if df.empty:
            # 注意：即使队伍历史数据为空，我们仍然可以为它计算赔率特征
            features = self._get_default_features(is_home)
            odds_features = self._calculate_odds_features(match_odds)
            features.update(odds_features)
            return features

        # 计算基础统计特征 (不变)
        features = {}
        features['avg_goals_scored'] = df['goals_for'].mean()
        features['avg_goals_conceded'] = df['goals_against'].mean()
        features['goals_scored_std'] = df['goals_for'].std() if len(df) > 1 else 0
        features['recent_form_goals'] = df['goals_for'].head(5).mean()
        features['current_elo'] = df['team_elo'].iloc[0]
        features['avg_opponent_elo'] = df['opponent_elo'].mean()
        features['elo_advantage'] = features['current_elo'] - features['avg_opponent_elo']
        
        home_matches = df[df['venue'] == 'home']
        away_matches = df[df['venue'] == 'away']
        
        if not home_matches.empty:
            features['home_goals_avg'] = home_matches['goals_for'].mean()
            features['home_goals_std'] = home_matches['goals_for'].std() if len(home_matches) > 1 else 0
        else:
            features['home_goals_avg'] = features['avg_goals_scored']
            features['home_goals_std'] = features['goals_scored_std']
            
        if not away_matches.empty:
            features['away_goals_avg'] = away_matches['goals_for'].mean()
            features['away_goals_std'] = away_matches['goals_for'].std() if len(away_matches) > 1 else 0
        else:
            features['away_goals_avg'] = features['avg_goals_scored']
            features['away_goals_std'] = features['goals_scored_std']
        
        features.update(self._calculate_elo_weighted_features(df))
        
        if not opponent_df.empty and opponent_df.iloc[0]['avg_goals_conceded'] is not None:
            features['opponent_defense_strength'] = opponent_df.iloc[0]['avg_goals_conceded']
            features['opponent_attack_strength'] = opponent_df.iloc[0]['avg_goals_scored']
        else:
            features['opponent_defense_strength'] = 1.5
            features['opponent_attack_strength'] = 1.5
        
        if len(df) >= 6:
            recent_6 = df.head(6)['goals_for'].mean()
            older_6 = df.tail(6)['goals_for'].mean() if len(df) >= 12 else recent_6
            features['goal_trend'] = recent_6 - older_6
        else:
            features['goal_trend'] = 0
        
        features['is_home'] = 1.0 if is_home else 0.0
        
        # 新增：计算并合并赔率特征
        odds_features = self._calculate_odds_features(match_odds)
        features.update(odds_features)
        
        return features

    def _calculate_elo_weighted_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算ELO加权特征"""
        if df.empty:
            return {'elo_weighted_goals': 1.5, 'elo_weight_sum': 0}
        
        current_elo = df.iloc[0]['team_elo']
        total_weighted_goals = 0
        total_weights = 0
        
        for i, row in df.iterrows():
            goals = row['goals_for']
            opponent_elo = row['opponent_elo']
            time_weight = 0.9 ** i
            elo_diff = abs(opponent_elo - current_elo)
            elo_weight = np.exp(-elo_diff / 200)
            combined_weight = time_weight * elo_weight
            total_weighted_goals += goals * combined_weight
            total_weights += combined_weight
        
        elo_weighted_goals = total_weighted_goals / total_weights if total_weights > 0 else df['goals_for'].mean()
        
        return {
            'elo_weighted_goals': elo_weighted_goals,
            'elo_weight_sum': total_weights
        }
    
    def _get_default_features(self, is_home: bool) -> Dict[str, float]:
        """获取默认特征值（已加入赔率特征的默认值）"""
        features = {
            'avg_goals_scored': 1.5, 'avg_goals_conceded': 1.5, 'goals_scored_std': 1.0,
            'recent_form_goals': 1.5, 'current_elo': 1500, 'avg_opponent_elo': 1500,
            'elo_advantage': 0, 'home_goals_avg': 1.6 if is_home else 1.4, 'home_goals_std': 1.0,
            'away_goals_avg': 1.4 if not is_home else 1.6, 'away_goals_std': 1.0,
            'elo_weighted_goals': 1.5, 'elo_weight_sum': 0, 'opponent_defense_strength': 1.5,
            'opponent_attack_strength': 1.5, 'goal_trend': 0, 'is_home': 1.0 if is_home else 0.0
        }
        # 为赔率特征设置默认值（-1表示数据缺失）
        odds_defaults = {
            'market_prob_h': -1.0, 'market_prob_d': -1.0, 'market_prob_a': -1.0,
            'market_prob_btts_yes': -1.0, 'market_prob_o15': -1.0, 'market_prob_o25': -1.0,
            'market_prob_o35': -1.0, 'market_implied_total_goals': -1.0
        }
        features.update(odds_defaults);
        return features
    
    def prepare_training_data(self, min_matches: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """准备训练数据（已修改为包含赔率）"""
        print("准备训练数据...")
        conn = sqlite3.connect(self.db_path)
        
        # 获取所有比赛数据，包括赔率
        query = f'''
            SELECT home_team, away_team, home_goals, away_goals, match_date, {', '.join(self.odds_cols)}
            FROM matches
            ORDER BY match_date
        '''
        matches_df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(matches_df) < min_matches:
            print(f"警告: 训练数据不足，当前只有 {len(matches_df)} 场")
            return pd.DataFrame(), pd.DataFrame()
        
        training_data = []
        for i, row in matches_df.iterrows():
            if i % 100 == 0:
                print(f"处理进度: {i}/{len(matches_df)}")
            
            home_team, away_team = row['home_team'], row['away_team']
            home_goals, away_goals = row['home_goals'], row['away_goals']
            
            # 提取当前比赛的赔率
            match_odds = {col: row[col] for col in self.odds_cols}
            
            try:
                # 提取主队特征
                home_features = self.extract_features(home_team, away_team, True, match_odds)
                home_features['target'] = home_goals
                home_features['team_type'] = 'home'
                training_data.append(home_features)
                
                # 提取客队特征
                away_features = self.extract_features(away_team, home_team, False, match_odds)
                away_features['target'] = away_goals
                away_features['team_type'] = 'away'
                training_data.append(away_features)
            except Exception as e:
                print(f"处理比赛时出错 {home_team} vs {away_team}: {e}")
                continue
        
        if not training_data:
            print("错误: 无法生成训练数据")
            return pd.DataFrame(), pd.DataFrame()
        
        training_df = pd.DataFrame(training_data).fillna(-1) # 用-1填充所有NaN
        
        home_data = training_df[training_df['team_type'] == 'home'].copy()
        away_data = training_df[training_df['team_type'] == 'away'].copy()
        
        feature_cols = [col for col in home_data.columns if col not in ['target', 'team_type']]
        home_X, home_y = home_data[feature_cols], home_data['target']
        away_X, away_y = away_data[feature_cols], away_data['target']
        
        print(f"训练数据准备完成: 主队 {len(home_X)} 样本, 客队 {len(away_X)} 样本")
        return (home_X, home_y), (away_X, away_y)
    
    def train_models(self):
        """训练机器学习模型"""
        print("开始训练机器学习模型...")
        (home_X, home_y), (away_X, away_y) = self.prepare_training_data()
        
        if home_X.empty or away_X.empty:
            print("错误: 训练数据为空")
            return False
        
        print("训练主队进球预测模型...")
        self.home_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
        X_train, X_test, y_train, y_test = train_test_split(home_X, home_y, test_size=0.2, random_state=42)
        self.home_model.fit(X_train, y_train)
        
        home_pred = self.home_model.predict(X_test)
        home_mae = mean_absolute_error(y_test, home_pred)
        print(f"主队模型性能: MAE={home_mae:.3f}")
        
        print("训练客队进球预测模型...")
        self.away_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
        X_train, X_test, y_train, y_test = train_test_split(away_X, away_y, test_size=0.2, random_state=42)
        self.away_model.fit(X_train, y_train)
        
        away_pred = self.away_model.predict(X_test)
        away_mae = mean_absolute_error(y_test, away_pred)
        print(f"客队模型性能: MAE={away_mae:.3f}")
        
        self.is_trained = True
        print("模型训练完成!")
        return True
    
    def predict_goals_ml(self, home_team: str, away_team: str, match_odds: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """使用机器学习模型预测进球数（可接收实时赔率）"""
        if not self.is_trained:
            print("警告: 模型未训练，正在自动训练...")
            if not self.train_models():
                return {'home_goals': 1.5, 'away_goals': 1.5, 'confidence': 0.0}
        
        if match_odds is None:
            print("警告: 未提供实时赔率，将使用默认值进行预测")
            match_odds = {col: 0 for col in self.odds_cols}

        try:
            home_features = self.extract_features(home_team, away_team, True, match_odds)
            away_features = self.extract_features(away_team, home_team, False, match_odds)
            
            home_df = pd.DataFrame([home_features]).fillna(-1)
            away_df = pd.DataFrame([away_features]).fillna(-1)
            
            feature_cols = self.home_model.feature_names_in_
            home_df = home_df[feature_cols]
            away_df = away_df[feature_cols]
            
            home_goals_pred = self.home_model.predict(home_df)[0]
            away_goals_pred = self.away_model.predict(away_df)[0]
            
            home_goals_pred = max(0.1, min(6.0, home_goals_pred))
            away_goals_pred = max(0.1, min(6.0, away_goals_pred))
            
            confidence = min(1.0, (home_features.get('elo_weight_sum', 0) + away_features.get('elo_weight_sum', 0)) / 20)
            
            return {
                'home_goals': round(home_goals_pred, 2),
                'away_goals': round(away_goals_pred, 2),
                'total_goals': round(home_goals_pred + away_goals_pred, 2),
                'confidence': round(confidence, 2)
            }
        except Exception as e:
            print(f"机器学习预测失败: {e}")
            return {'home_goals': 1.5, 'away_goals': 1.5, 'total_goals': 3.0, 'confidence': 0.0}

if __name__ == "__main__":
    predictor = AdvancedGoalPredictor()
    predictor.train_models()
    
    # 预测示例（不带赔率）
    print("\n--- 预测示例 (不带赔率) ---")
    prediction = predictor.predict_goals_ml("manchester city", "arsenal")
    print(f"预测结果: {prediction}")

    # 预测示例（带实时赔率）
    print("\n--- 预测示例 (带实时赔率) ---")
    example_odds = {
        'odds_ft_home_team_win': 1.8, 'odds_ft_draw': 3.5, 'odds_ft_away_team_win': 4.0,
        'odds_ft_over15': 1.2, 'odds_ft_over25': 1.7, 'odds_ft_over35': 2.5,
        'odds_btts_yes': 1.6, 'odds_btts_no': 2.2
    }
    prediction_with_odds = predictor.predict_goals_ml("manchester city", "arsenal", match_odds=example_odds)
    print(f"带赔率预测结果: {prediction_with_odds}")