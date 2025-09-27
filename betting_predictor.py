#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实用的足球博彩预测接口
基于训练好的自适应权重模型进行预测和投注建议
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from adaptive_betting_system import AdaptiveBettingSystem

class BettingPredictor:
    """博彩预测器 - 实用接口"""
    
    def __init__(self):
        self.system = AdaptiveBettingSystem()
        self.is_initialized = False
        
    def initialize(self):
        """初始化系统（只需运行一次）"""
        if not self.is_initialized:
            print("🚀 初始化预测系统...")
            self.system.initialize_system()
            self.is_initialized = True
            print("✅ 系统初始化完成！")
    
    def predict_match(self, home_team: str, away_team: str, 
                     over15_odds: float = 1.2, over25_odds: float = 1.8, 
                     over35_odds: float = 3.2, over45_odds: float = 6.5) -> Dict:
        """
        预测单场比赛
        
        Args:
            home_team: 主队名称
            away_team: 客队名称
            over15_odds: Over 1.5 赔率
            over25_odds: Over 2.5 赔率
            over35_odds: Over 3.5 赔率
            over45_odds: Over 4.5 赔率
        
        Returns:
            预测结果和投注建议
        """
        if not self.is_initialized:
            self.initialize()
        
        # 构造市场赔率
        market_odds = {
            'odds_ft_over15': over15_odds,
            'odds_ft_over25': over25_odds,
            'odds_ft_over35': over35_odds,
            'odds_ft_over45': over45_odds
        }
        
        # 获取预测结果
        result = self.system.predict_and_recommend(home_team, away_team, market_odds)
        
        # 格式化输出
        prediction = result['prediction']
        
        formatted_result = {
            'match_info': {
                'home_team': home_team.title(),
                'away_team': away_team.title(),
                'predicted_total_goals': round(prediction['predicted_total'], 2)
            },
            'probabilities': {
                'over_15': f"{prediction['probabilities']['over_15']:.1%}",
                'over_25': f"{prediction['probabilities']['over_25']:.1%}",
                'over_35': f"{prediction['probabilities']['over_35']:.1%}",
                'over_45': f"{prediction['probabilities']['over_45']:.1%}",
                'under_25': f"{prediction['probabilities']['under_25']:.1%}",
                'under_35': f"{prediction['probabilities']['under_35']:.1%}"
            },
            'model_details': {
                'elo_prediction': round(prediction['component_predictions']['elo'], 2),
                'ml_prediction': round(prediction['component_predictions']['ml'], 2),
                'market_adjustment': round(prediction['component_predictions']['market'], 2),
                'weights': {k: f"{v:.1%}" for k, v in prediction['weights'].items()},
                'confidence': f"{prediction['confidence']:.1%}"
            },
            'betting_advice': {
                'recommendation': result['recommendation'],
                'value_bets': []
            }
        }
        
        # 添加价值投注详情
        for bet in result['value_bets']:
            formatted_result['betting_advice']['value_bets'].append({
                'bet_type': bet['bet_type'].replace('_', ' ').title(),
                'expected_value': f"{bet['expected_value']:.1%}",
                'our_probability': f"{bet['our_probability']:.1%}",
                'market_odds': bet['market_odds'],
                'kelly_fraction': f"{bet['kelly_fraction']:.1%}",
                'confidence': f"{bet['confidence']:.1%}"
            })
        
        return formatted_result
    
    def analyze_multiple_matches(self, matches: List[Dict]) -> pd.DataFrame:
        """
        分析多场比赛
        
        Args:
            matches: 比赛列表，每个元素包含 home_team, away_team, 和赔率信息
        
        Returns:
            分析结果DataFrame
        """
        results = []
        
        for match in matches:
            try:
                result = self.predict_match(
                    match['home_team'], 
                    match['away_team'],
                    match.get('over15_odds', 1.2),
                    match.get('over25_odds', 1.8),
                    match.get('over35_odds', 3.2),
                    match.get('over45_odds', 6.5)
                )
                
                # 提取关键信息
                best_bet = None
                max_ev = 0
                
                for bet in result['betting_advice']['value_bets']:
                    ev = float(bet['expected_value'].strip('%')) / 100
                    if ev > max_ev:
                        max_ev = ev
                        best_bet = bet
                
                results.append({
                    'match': f"{result['match_info']['home_team']} vs {result['match_info']['away_team']}",
                    'predicted_goals': result['match_info']['predicted_total_goals'],
                    'over_25_prob': result['probabilities']['over_25'],
                    'confidence': result['model_details']['confidence'],
                    'best_bet': best_bet['bet_type'] if best_bet else 'None',
                    'expected_value': f"{max_ev:.1%}" if best_bet else '0.0%',
                    'recommendation': 'BET' if best_bet and max_ev >= 0.03 else 'PASS'
                })
                
            except Exception as e:
                print(f"分析失败 {match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def get_daily_recommendations(self, matches: List[Dict], min_ev: float = 0.03) -> List[Dict]:
        """
        获取每日投注建议
        
        Args:
            matches: 当日比赛列表
            min_ev: 最小期望收益阈值
        
        Returns:
            推荐投注列表
        """
        recommendations = []
        
        for match in matches:
            result = self.predict_match(
                match['home_team'], 
                match['away_team'],
                match.get('over15_odds', 1.2),
                match.get('over25_odds', 1.8),
                match.get('over35_odds', 3.2),
                match.get('over45_odds', 6.5)
            )
            
            # 筛选高价值投注
            for bet in result['betting_advice']['value_bets']:
                ev = float(bet['expected_value'].strip('%')) / 100
                if ev >= min_ev:
                    recommendations.append({
                        'match': f"{result['match_info']['home_team']} vs {result['match_info']['away_team']}",
                        'bet_type': bet['bet_type'],
                        'expected_value': bet['expected_value'],
                        'odds': bet['market_odds'],
                        'kelly_stake': bet['kelly_fraction'],
                        'confidence': bet['confidence'],
                        'priority': 'HIGH' if ev >= 0.05 else 'MEDIUM'
                    })
        
        # 按期望收益排序
        recommendations.sort(key=lambda x: float(x['expected_value'].strip('%')), reverse=True)
        
        return recommendations[:10]  # 返回前10个最佳机会

def print_prediction_report(result: Dict):
    """打印格式化的预测报告"""
    print("=" * 80)
    print(f"🏟️  {result['match_info']['home_team']} vs {result['match_info']['away_team']}")
    print("=" * 80)
    
    print(f"\n📊 预测结果:")
    print(f"   总进球数: {result['match_info']['predicted_total_goals']}")
    print(f"   置信度: {result['model_details']['confidence']}")
    
    print(f"\n🎯 各档位概率:")
    probs = result['probabilities']
    print(f"   Over 1.5: {probs['over_15']} | Over 2.5: {probs['over_25']} | Over 3.5: {probs['over_35']}")
    print(f"   Under 2.5: {probs['under_25']} | Under 3.5: {probs['under_35']}")
    
    print(f"\n🤖 模型详情:")
    details = result['model_details']
    print(f"   ELO预测: {details['elo_prediction']} (权重: {details['weights']['elo']})")
    print(f"   ML预测: {details['ml_prediction']} (权重: {details['weights']['ml']})")
    print(f"   市场校正: {details['market_adjustment']} (权重: {details['weights']['market']})")
    
    print(f"\n💰 投注建议:")
    if result['betting_advice']['value_bets']:
        for bet in result['betting_advice']['value_bets']:
            print(f"   ✅ {bet['bet_type']} @{bet['market_odds']:.2f}")
            print(f"      期望收益: {bet['expected_value']} | 建议投注: {bet['kelly_fraction']}")
    else:
        print("   ❌ 无价值投注机会")
    
    print("=" * 80)

# 使用示例
if __name__ == "__main__":
    # 创建预测器
    predictor = BettingPredictor()
    
    print("🎯 足球博彩智能预测系统")
    print("基于K-fold训练的自适应权重模型")
    print("最优权重: ELO 10%, ML 60%, Market 30%")
    print("预期收益率: 22.33%\n")
    
    # 示例1: 单场比赛预测
    print("=" * 50)
    print("示例1: 单场比赛预测")
    print("=" * 50)
    
    result = predictor.predict_match(
        home_team="manchester city",
        away_team="arsenal", 
        over15_odds=1.15,
        over25_odds=1.75,
        over35_odds=3.0,
        over45_odds=6.0
    )
    
    print_prediction_report(result)
    
    # 示例2: 多场比赛分析
    print("\n" + "=" * 50)
    print("示例2: 多场比赛分析")
    print("=" * 50)
    
    sample_matches = [
        {
            'home_team': 'liverpool',
            'away_team': 'chelsea',
            'over15_odds': 1.1,
            'over25_odds': 1.6,
            'over35_odds': 2.8,
            'over45_odds': 5.5
        },
        {
            'home_team': 'real madrid',
            'away_team': 'barcelona',
            'over15_odds': 1.2,
            'over25_odds': 1.9,
            'over35_odds': 3.5,
            'over45_odds': 7.0
        }
    ]
    
    analysis_df = predictor.analyze_multiple_matches(sample_matches)
    print(analysis_df.to_string(index=False))
    
    # 示例3: 每日推荐
    print("\n" + "=" * 50)
    print("示例3: 每日投注推荐")
    print("=" * 50)
    
    daily_recommendations = predictor.get_daily_recommendations(sample_matches, min_ev=0.02)
    
    if daily_recommendations:
        print("🏆 今日推荐投注:")
        for i, rec in enumerate(daily_recommendations, 1):
            print(f"{i}. {rec['match']} - {rec['bet_type']}")
            print(f"   期望收益: {rec['expected_value']} | 赔率: {rec['odds']:.2f} | 优先级: {rec['priority']}")
    else:
        print("❌ 今日无推荐投注")
    
    print(f"\n🎯 系统已就绪，可以开始实际预测！")