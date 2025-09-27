#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应权重的足球博彩预测系统
基于K-fold交叉验证训练最优权重分配
专注于总进球市场的期望收益最大化
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from football_betting_system import FootballBettingSystem
from advanced_goal_prediction import AdvancedGoalPredictor

class AdaptiveWeightModel:
    """自适应权重学习模型"""
    
    def __init__(self):
        self.weights = {
            'elo': 0.33,
            'ml': 0.33,
            'market': 0.34
        }
        self.learning_rate = 0.005
        self.performance_history = []
        self.optimization_window = 100
        
    def update_weights(self, elo_performance: float, ml_performance: float, market_performance: float):
        """基于各预测源的表现更新权重"""
        total_performance = elo_performance + ml_performance + market_performance
        
        if total_performance > 0:
            # 计算新权重
            new_weights = {
                'elo': elo_performance / total_performance,
                'ml': ml_performance / total_performance,
                'market': market_performance / total_performance
            }
            
            # 平滑更新
            for key in self.weights:
                self.weights[key] = (1 - self.learning_rate) * self.weights[key] + \
                                  self.learning_rate * new_weights[key]
    
    def get_weighted_prediction(self, elo_pred: float, ml_pred: float, market_pred: float) -> float:
        """获取加权预测结果"""
        return (self.weights['elo'] * elo_pred + 
                self.weights['ml'] * ml_pred + 
                self.weights['market'] * market_pred)

class TotalGoalsPredictor:
    """总进球数预测器"""
    
    def __init__(self, db_path: str = "football_betting.db"):
        self.db_path = db_path
        self.base_system = FootballBettingSystem(db_path=db_path)
        self.ml_predictor = AdvancedGoalPredictor(db_path=db_path)
        self.weight_model = AdaptiveWeightModel()
        
    def calculate_over_probability(self, predicted_total: float, threshold: float) -> float:
        """计算超过某个进球数的概率（泊松分布）"""
        def poisson_cdf(k, lam):
            if lam <= 0:
                return 1.0
            prob_sum = 0
            for i in range(int(k) + 1):
                prob_sum += (lam ** i) * np.exp(-lam) / np.math.factorial(i)
            return min(1.0, prob_sum)
        
        prob_over = 1 - poisson_cdf(threshold, predicted_total)
        return max(0.01, min(0.99, prob_over))
    
    def get_market_implied_probability(self, odds: float) -> float:
        """从赔率获取市场隐含概率"""
        if odds <= 1.0:
            return 0.0
        return 1.0 / odds
    
    def predict_total_goals(self, home_team: str, away_team: str) -> Dict:
        """预测总进球数（三层融合）"""
        try:
            # 第一层：ELO预测
            elo_prediction = self.base_system.predict_match_goals(home_team, away_team)
            elo_total = elo_prediction['total_goals_predicted']
            
            # 第二层：机器学习预测
            ml_prediction = self.ml_predictor.predict_goals_ml(home_team, away_team)
            ml_total = ml_prediction['total_goals']
            
            # 第三层：市场校正（暂时使用历史平均，后续会用实际赔率）
            market_total = (elo_total + ml_total) / 2  # 简化的市场校正
            
            # 加权融合
            final_prediction = self.weight_model.get_weighted_prediction(
                elo_total, ml_total, market_total
            )
            
            # 计算各档位概率
            probabilities = {
                'over_15': self.calculate_over_probability(final_prediction, 1.5),
                'over_25': self.calculate_over_probability(final_prediction, 2.5),
                'over_35': self.calculate_over_probability(final_prediction, 3.5),
                'over_45': self.calculate_over_probability(final_prediction, 4.5)
            }
            
            # 计算Under概率
            probabilities.update({
                'under_15': 1 - probabilities['over_15'],
                'under_25': 1 - probabilities['over_25'],
                'under_35': 1 - probabilities['over_35'],
                'under_45': 1 - probabilities['over_45']
            })
            
            return {
                'predicted_total': final_prediction,
                'probabilities': probabilities,
                'component_predictions': {
                    'elo': elo_total,
                    'ml': ml_total,
                    'market': market_total
                },
                'weights': self.weight_model.weights.copy(),
                'confidence': (elo_prediction['confidence'] + ml_prediction['confidence']) / 2
            }
            
        except Exception as e:
            print(f"预测失败 {home_team} vs {away_team}: {e}")
            return self._get_default_prediction()
    
    def _get_default_prediction(self) -> Dict:
        """获取默认预测"""
        default_total = 2.5
        return {
            'predicted_total': default_total,
            'probabilities': {
                'over_15': 0.8, 'under_15': 0.2,
                'over_25': 0.5, 'under_25': 0.5,
                'over_35': 0.3, 'under_35': 0.7,
                'over_45': 0.15, 'under_45': 0.85
            },
            'component_predictions': {'elo': default_total, 'ml': default_total, 'market': default_total},
            'weights': self.weight_model.weights.copy(),
            'confidence': 0.3
        }

class ExpectedValueCalculator:
    """期望收益计算器"""
    
    def __init__(self):
        self.min_expected_value = 0.03  # 最小3%期望收益
        self.min_confidence = 0.4       # 最小40%置信度
        self.max_odds = 8.0            # 最大赔率限制
        self.min_odds = 1.2            # 最小赔率限制
        
    def calculate_expected_value(self, our_probability: float, odds: float) -> float:
        """计算期望收益"""
        if odds <= 1.0:
            return -1.0
        return (our_probability * odds) - 1.0
    
    def find_value_bets(self, prediction: Dict, market_odds: Dict) -> List[Dict]:
        """寻找价值投注机会"""
        value_bets = []
        
        if prediction['confidence'] < self.min_confidence:
            return value_bets
        
        # 检查各个总进球档位
        bet_types = ['over_15', 'over_25', 'over_35', 'over_45', 
                    'under_15', 'under_25', 'under_35', 'under_45']
        
        for bet_type in bet_types:
            # 获取我们的概率预测
            our_prob = prediction['probabilities'].get(bet_type, 0)
            
            # 获取市场赔率
            odds_key = f"odds_ft_{bet_type.replace('_', '')}"
            if bet_type.startswith('under'):
                # Under赔率需要从Over赔率推算
                over_key = f"odds_ft_{bet_type.replace('under_', 'over')}"
                over_odds = market_odds.get(over_key, 0)
                if over_odds > 1:
                    over_prob = 1 / over_odds
                    under_prob = 1 - over_prob
                    odds = 1 / under_prob if under_prob > 0.01 else 100
                else:
                    continue
            else:
                odds = market_odds.get(odds_key, 0)
            
            if not (self.min_odds <= odds <= self.max_odds):
                continue
            
            # 计算期望收益
            expected_value = self.calculate_expected_value(our_prob, odds)
            
            if expected_value >= self.min_expected_value:
                value_bets.append({
                    'bet_type': bet_type,
                    'our_probability': our_prob,
                    'market_odds': odds,
                    'expected_value': expected_value,
                    'confidence': prediction['confidence'],
                    'kelly_fraction': self._calculate_kelly_fraction(our_prob, odds)
                })
        
        return value_bets
    
    def _calculate_kelly_fraction(self, win_prob: float, odds: float) -> float:
        """计算凯利公式投注比例"""
        if odds <= 1 or win_prob <= 0:
            return 0
        
        b = odds - 1
        p = win_prob
        q = 1 - p
        
        kelly = (b * p - q) / b
        conservative_kelly = kelly * 0.25  # 25%保守系数
        
        return max(0, min(0.02, conservative_kelly))  # 最大2%

class KFoldTrainer:
    """K-fold交叉验证训练器"""
    
    def __init__(self, predictor: TotalGoalsPredictor, k_folds: int = 5):
        self.predictor = predictor
        self.k_folds = k_folds
        self.best_weights = None
        self.best_performance = -np.inf
        
    def prepare_training_data(self) -> pd.DataFrame:
        """准备训练数据"""
        print("准备K-fold训练数据...")
        
        # 从CSV文件加载包含赔率的数据
        all_data = []
        leagues = [
            ("england", "premier-league"),
            ("germany", "bundesliga"),
            ("spain", "la-liga"),
            ("france", "ligue-1"),
            ("italy", "serie-a")
        ]
        
        for country, league in leagues:
            for season in ["2022-to-2023", "2023-to-2024"]:
                file_path = f"data/{country}-{league}-matches-{season}-stats.csv"
                try:
                    df = pd.read_csv(file_path)
                    # 过滤有效数据
                    df = df.dropna(subset=['home_team_name', 'away_team_name', 
                                         'home_team_goal_count', 'away_team_goal_count'])
                    df = df[df['odds_ft_over25'] > 1]  # 确保有有效赔率
                    df['league'] = f"{country}-{league}"
                    all_data.append(df)
                    print(f"加载 {country}-{league}-{season}: {len(df)} 场比赛")
                except FileNotFoundError:
                    continue
        
        if not all_data:
            raise ValueError("没有找到训练数据")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"总训练数据: {len(combined_df)} 场比赛")
        
        return combined_df
    
    def evaluate_weights(self, weights: Dict, test_data: pd.DataFrame) -> float:
        """评估权重组合的性能"""
        # 临时设置权重
        original_weights = self.predictor.weight_model.weights.copy()
        self.predictor.weight_model.weights = weights
        
        total_return = 0
        total_bets = 0
        ev_calculator = ExpectedValueCalculator()
        
        for _, match in test_data.iterrows():
            try:
                home_team = match['home_team_name'].strip().lower()
                away_team = match['away_team_name'].strip().lower()
                actual_total = match['total_goal_count']
                
                # 获取预测
                prediction = self.predictor.predict_total_goals(home_team, away_team)
                
                # 获取市场赔率
                market_odds = {
                    'odds_ft_over15': match.get('odds_ft_over15', 0),
                    'odds_ft_over25': match.get('odds_ft_over25', 0),
                    'odds_ft_over35': match.get('odds_ft_over35', 0),
                    'odds_ft_over45': match.get('odds_ft_over45', 0)
                }
                
                # 寻找价值投注
                value_bets = ev_calculator.find_value_bets(prediction, market_odds)
                
                for bet in value_bets:
                    total_bets += 1
                    bet_won = self._check_bet_result(actual_total, bet['bet_type'])
                    
                    if bet_won:
                        total_return += bet['market_odds'] - 1  # 净收益
                    else:
                        total_return -= 1  # 损失本金
                        
            except Exception:
                continue
        
        # 恢复原权重
        self.predictor.weight_model.weights = original_weights
        
        # 计算平均收益率
        if total_bets > 0:
            return total_return / total_bets
        else:
            return -1.0
    
    def _check_bet_result(self, actual_total: int, bet_type: str) -> bool:
        """检查投注结果"""
        if bet_type == 'over_15':
            return actual_total > 1.5
        elif bet_type == 'over_25':
            return actual_total > 2.5
        elif bet_type == 'over_35':
            return actual_total > 3.5
        elif bet_type == 'over_45':
            return actual_total > 4.5
        elif bet_type == 'under_15':
            return actual_total < 1.5
        elif bet_type == 'under_25':
            return actual_total < 2.5
        elif bet_type == 'under_35':
            return actual_total < 3.5
        elif bet_type == 'under_45':
            return actual_total < 4.5
        return False
    
    def train_optimal_weights(self) -> Dict:
        """使用K-fold交叉验证训练最优权重"""
        print(f"开始{self.k_folds}-fold交叉验证训练...")
        
        # 准备数据
        training_data = self.prepare_training_data()
        
        # K-fold分割
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        
        # 权重搜索空间
        weight_combinations = []
        for elo_w in np.arange(0.1, 0.7, 0.1):
            for ml_w in np.arange(0.1, 0.7, 0.1):
                market_w = 1.0 - elo_w - ml_w
                if 0.1 <= market_w <= 0.7:
                    weight_combinations.append({
                        'elo': elo_w,
                        'ml': ml_w,
                        'market': market_w
                    })
        
        print(f"测试 {len(weight_combinations)} 种权重组合...")
        
        best_avg_performance = -np.inf
        best_weights = None
        
        for i, weights in enumerate(weight_combinations):
            if i % 10 == 0:
                print(f"进度: {i}/{len(weight_combinations)}")
            
            fold_performances = []
            
            # K-fold验证
            for fold, (train_idx, test_idx) in enumerate(kf.split(training_data)):
                test_fold = training_data.iloc[test_idx]
                performance = self.evaluate_weights(weights, test_fold)
                fold_performances.append(performance)
            
            # 计算平均性能
            avg_performance = np.mean(fold_performances)
            
            if avg_performance > best_avg_performance:
                best_avg_performance = avg_performance
                best_weights = weights.copy()
                print(f"新的最佳权重: {weights}, 平均收益率: {avg_performance:.4f}")
        
        self.best_weights = best_weights
        self.best_performance = best_avg_performance
        
        # 应用最佳权重
        if best_weights:
            self.predictor.weight_model.weights = best_weights
            print(f"\n=== 训练完成 ===")
            print(f"最佳权重: {best_weights}")
            print(f"预期收益率: {best_avg_performance:.4f}")
        
        return best_weights

class AdaptiveBettingSystem:
    """自适应博彩系统主类"""
    
    def __init__(self, db_path: str = "football_betting.db"):
        self.db_path = db_path
        self.predictor = TotalGoalsPredictor(db_path)
        self.ev_calculator = ExpectedValueCalculator()
        self.trainer = KFoldTrainer(self.predictor)
        
    def initialize_system(self):
        """初始化系统"""
        print("=== 初始化自适应博彩系统 ===")
        
        # 初始化基础系统
        print("1. 初始化基础预测系统...")
        self.predictor.base_system.run_full_analysis()
        
        # 训练机器学习模型
        print("2. 训练机器学习模型...")
        self.predictor.ml_predictor.train_models()
        
        # K-fold训练最优权重
        print("3. K-fold训练最优权重...")
        best_weights = self.trainer.train_optimal_weights()
        
        return best_weights
    
    def predict_and_recommend(self, home_team: str, away_team: str, market_odds: Dict) -> Dict:
        """预测并生成投注建议"""
        # 获取预测
        prediction = self.predictor.predict_total_goals(home_team, away_team)
        
        # 寻找价值投注
        value_bets = self.ev_calculator.find_value_bets(prediction, market_odds)
        
        return {
            'match': f"{home_team} vs {away_team}",
            'prediction': prediction,
            'value_bets': value_bets,
            'recommendation': self._generate_recommendation(value_bets)
        }
    
    def _generate_recommendation(self, value_bets: List[Dict]) -> str:
        """生成投注建议"""
        if not value_bets:
            return "无价值投注机会"
        
        # 选择期望收益最高的投注
        best_bet = max(value_bets, key=lambda x: x['expected_value'])
        
        return f"推荐投注: {best_bet['bet_type']} @{best_bet['market_odds']:.2f}, " \
               f"期望收益: {best_bet['expected_value']:.2%}, " \
               f"建议投注比例: {best_bet['kelly_fraction']:.2%}"

# 使用示例
if __name__ == "__main__":
    # 创建系统
    system = AdaptiveBettingSystem()
    
    # 初始化和训练
    best_weights = system.initialize_system()
    
    # 示例预测
    if best_weights:
        print("\n=== 预测示例 ===")
        
        # 模拟市场赔率
        sample_odds = {
            'odds_ft_over15': 1.2,
            'odds_ft_over25': 1.8,
            'odds_ft_over35': 3.2,
            'odds_ft_over45': 6.5
        }
        
        try:
            result = system.predict_and_recommend("manchester city", "arsenal", sample_odds)
            
            print(f"比赛: {result['match']}")
            print(f"预测总进球: {result['prediction']['predicted_total']:.2f}")
            print(f"当前权重: {result['prediction']['weights']}")
            print(f"置信度: {result['prediction']['confidence']:.2f}")
            print(f"投注建议: {result['recommendation']}")
            
            if result['value_bets']:
                print("\n价值投注详情:")
                for bet in result['value_bets']:
                    print(f"  {bet['bet_type']}: EV={bet['expected_value']:.2%}, "
                          f"概率={bet['our_probability']:.2%}, 赔率={bet['market_odds']:.2f}")
        
        except Exception as e:
            print(f"预测示例失败: {e}")
    
    print("\n🎯 系统训练完成，可以开始使用！")