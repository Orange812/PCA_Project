#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
足球博彩建议系统
整合ELO评分、机器学习预测和博彩建议
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from football_betting_system import FootballBettingSystem
from advanced_goal_prediction import AdvancedGoalPredictor

class BettingAdvisor:
    """足球博彩建议系统"""
    
    def __init__(self, db_path: str = "football_betting.db"):
        self.betting_system = FootballBettingSystem(db_path=db_path)
        self.goal_predictor = AdvancedGoalPredictor(db_path=db_path)
        self.confidence_threshold = 0.6  # 建议的最低置信度阈值
        
    def get_comprehensive_prediction(self, home_team: str, away_team: str, 
                                   recent_matches: int = 12) -> Dict:
        """获取综合预测结果"""
        print(f"\n=== 分析比赛: {home_team} vs {away_team} ===")
        
        # 1. ELO基础预测
        elo_prediction = self.betting_system.predict_match_goals(home_team, away_team, recent_matches)
        
        # 2. 机器学习预测
        ml_prediction = self.goal_predictor.predict_goals_ml(home_team, away_team)
        
        # 3. 综合预测（加权平均）
        elo_weight = 0.4
        ml_weight = 0.6
        
        combined_home_goals = (elo_prediction['home_goals_predicted'] * elo_weight + 
                             ml_prediction['home_goals'] * ml_weight)
        combined_away_goals = (elo_prediction['away_goals_predicted'] * elo_weight + 
                             ml_prediction['away_goals'] * ml_weight)
        combined_total_goals = combined_home_goals + combined_away_goals
        
        # 4. 综合置信度
        combined_confidence = (elo_prediction['confidence'] * elo_weight + 
                             ml_prediction['confidence'] * ml_weight)
        
        return {
            'match': f"{home_team} vs {away_team}",
            'elo_prediction': elo_prediction,
            'ml_prediction': ml_prediction,
            'combined_prediction': {
                'home_goals': round(combined_home_goals, 2),
                'away_goals': round(combined_away_goals, 2),
                'total_goals': round(combined_total_goals, 2),
                'confidence': round(combined_confidence, 2)
            },
            'home_win_prob': elo_prediction['home_win_probability'],
            'draw_prob': elo_prediction['draw_probability'],
            'away_win_prob': elo_prediction['away_win_probability']
        }
    
    def generate_betting_suggestions(self, prediction: Dict) -> List[Dict]:
        """生成博彩建议"""
        suggestions = []
        combined = prediction['combined_prediction']
        confidence = combined['confidence']
        
        if confidence < self.confidence_threshold:
            suggestions.append({
                'type': '风险提醒',
                'suggestion': f'预测置信度较低 ({confidence:.2f})，建议谨慎投注',
                'risk_level': 'HIGH'
            })
            return suggestions
        
        home_goals = combined['home_goals']
        away_goals = combined['away_goals']
        total_goals = combined['total_goals']
        
        # 1. 总进球数建议
        if total_goals >= 3.5:
            suggestions.append({
                'type': '总进球数',
                'suggestion': f'大球 (Over 3.5) - 预测总进球 {total_goals}',
                'confidence': confidence,
                'risk_level': 'MEDIUM'
            })
        elif total_goals >= 2.5:
            suggestions.append({
                'type': '总进球数',
                'suggestion': f'大球 (Over 2.5) - 预测总进球 {total_goals}',
                'confidence': confidence,
                'risk_level': 'LOW'
            })
        elif total_goals <= 1.5:
            suggestions.append({
                'type': '总进球数',
                'suggestion': f'小球 (Under 1.5) - 预测总进球 {total_goals}',
                'confidence': confidence,
                'risk_level': 'MEDIUM'
            })
        else:
            suggestions.append({
                'type': '总进球数',
                'suggestion': f'小球 (Under 2.5) - 预测总进球 {total_goals}',
                'confidence': confidence,
                'risk_level': 'LOW'
            })
        
        # 2. 胜负建议
        home_prob = prediction['home_win_prob']
        draw_prob = prediction['draw_prob']
        away_prob = prediction['away_win_prob']
        
        max_prob = max(home_prob, draw_prob, away_prob)
        
        if max_prob == home_prob and home_prob > 0.5:
            suggestions.append({
                'type': '胜负',
                'suggestion': f'主胜 - 概率 {home_prob:.3f}',
                'confidence': confidence,
                'risk_level': 'LOW' if home_prob > 0.6 else 'MEDIUM'
            })
        elif max_prob == away_prob and away_prob > 0.5:
            suggestions.append({
                'type': '胜负',
                'suggestion': f'客胜 - 概率 {away_prob:.3f}',
                'confidence': confidence,
                'risk_level': 'LOW' if away_prob > 0.6 else 'MEDIUM'
            })
        else:
            suggestions.append({
                'type': '胜负',
                'suggestion': '比赛结果难以预测，建议避免胜负投注',
                'confidence': confidence,
                'risk_level': 'HIGH'
            })
        
        # 3. 双方进球建议
        if home_goals >= 1.0 and away_goals >= 1.0:
            suggestions.append({
                'type': '双方进球',
                'suggestion': f'双方进球 (BTTS Yes) - 主队 {home_goals}, 客队 {away_goals}',
                'confidence': confidence,
                'risk_level': 'LOW'
            })
        else:
            suggestions.append({
                'type': '双方进球',
                'suggestion': f'单方进球 (BTTS No) - 主队 {home_goals}, 客队 {away_goals}',
                'confidence': confidence,
                'risk_level': 'MEDIUM'
            })
        
        # 4. 半场/全场建议
        if abs(home_goals - away_goals) > 1.5:
            stronger_team = "主队" if home_goals > away_goals else "客队"
            suggestions.append({
                'type': '半场/全场',
                'suggestion': f'{stronger_team}实力明显更强，可考虑半场/全场投注',
                'confidence': confidence,
                'risk_level': 'MEDIUM'
            })
        
        return suggestions
    
    def analyze_multiple_matches(self, matches: List[Tuple[str, str]]) -> pd.DataFrame:
        """分析多场比赛"""
        results = []
        
        for home_team, away_team in matches:
            try:
                prediction = self.get_comprehensive_prediction(home_team, away_team)
                suggestions = self.generate_betting_suggestions(prediction)
                
                # 提取关键信息
                combined = prediction['combined_prediction']
                
                result = {
                    'match': f"{home_team} vs {away_team}",
                    'predicted_score': f"{combined['home_goals']:.1f} - {combined['away_goals']:.1f}",
                    'total_goals': combined['total_goals'],
                    'confidence': combined['confidence'],
                    'home_win_prob': prediction['home_win_prob'],
                    'draw_prob': prediction['draw_prob'],
                    'away_win_prob': prediction['away_win_prob'],
                    'main_suggestion': suggestions[0]['suggestion'] if suggestions else 'No suggestion',
                    'risk_level': suggestions[0]['risk_level'] if suggestions else 'UNKNOWN'
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"分析失败 {home_team} vs {away_team}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def print_detailed_analysis(self, home_team: str, away_team: str):
        """打印详细分析报告"""
        prediction = self.get_comprehensive_prediction(home_team, away_team)
        suggestions = self.generate_betting_suggestions(prediction)
        
        print(f"\n{'='*60}")
        print(f"详细分析报告: {home_team.upper()} vs {away_team.upper()}")
        print(f"{'='*60}")
        
        # ELO预测
        elo = prediction['elo_prediction']
        print(f"\n📊 ELO评分预测:")
        print(f"   主队ELO: {elo['home_elo']:.0f}")
        print(f"   客队ELO: {elo['away_elo']:.0f}")
        print(f"   预测比分: {elo['home_goals_predicted']:.1f} - {elo['away_goals_predicted']:.1f}")
        print(f"   置信度: {elo['confidence']:.2f}")
        
        # 机器学习预测
        ml = prediction['ml_prediction']
        print(f"\n🤖 机器学习预测:")
        print(f"   预测比分: {ml['home_goals']:.1f} - {ml['away_goals']:.1f}")
        print(f"   总进球数: {ml['total_goals']:.1f}")
        print(f"   置信度: {ml['confidence']:.2f}")
        
        # 综合预测
        combined = prediction['combined_prediction']
        print(f"\n🎯 综合预测:")
        print(f"   最终比分: {combined['home_goals']:.1f} - {combined['away_goals']:.1f}")
        print(f"   总进球数: {combined['total_goals']:.1f}")
        print(f"   综合置信度: {combined['confidence']:.2f}")
        
        # 胜负概率
        print(f"\n📈 胜负概率:")
        print(f"   主胜: {prediction['home_win_prob']:.1%}")
        print(f"   平局: {prediction['draw_prob']:.1%}")
        print(f"   客胜: {prediction['away_win_prob']:.1%}")
        
        # 博彩建议
        print(f"\n💡 博彩建议:")
        for i, suggestion in enumerate(suggestions, 1):
            risk_emoji = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🔴'}.get(suggestion['risk_level'], '⚪')
            print(f"   {i}. [{suggestion['type']}] {suggestion['suggestion']}")
            print(f"      风险等级: {risk_emoji} {suggestion['risk_level']}")
        
        print(f"\n{'='*60}")
    
    def get_best_bets_today(self, matches: List[Tuple[str, str]], min_confidence: float = 0.7) -> List[Dict]:
        """获取今日最佳投注建议"""
        print(f"\n🔍 寻找最佳投注机会 (最低置信度: {min_confidence})")
        
        best_bets = []
        
        for home_team, away_team in matches:
            try:
                prediction = self.get_comprehensive_prediction(home_team, away_team)
                suggestions = self.generate_betting_suggestions(prediction)
                
                confidence = prediction['combined_prediction']['confidence']
                
                if confidence >= min_confidence:
                    for suggestion in suggestions:
                        if suggestion['risk_level'] in ['LOW', 'MEDIUM']:
                            best_bets.append({
                                'match': f"{home_team} vs {away_team}",
                                'suggestion': suggestion['suggestion'],
                                'type': suggestion['type'],
                                'confidence': confidence,
                                'risk_level': suggestion['risk_level'],
                                'predicted_score': f"{prediction['combined_prediction']['home_goals']:.1f} - {prediction['combined_prediction']['away_goals']:.1f}"
                            })
            except Exception as e:
                continue
        
        # 按置信度排序
        best_bets.sort(key=lambda x: x['confidence'], reverse=True)
        
        return best_bets[:10]  # 返回前10个最佳建议

# 使用示例
if __name__ == "__main__":
    # 创建博彩建议系统
    advisor = BettingAdvisor()
    
    # 确保系统已初始化
    print("初始化博彩分析系统...")
    advisor.betting_system.run_full_analysis()
    
    # 训练机器学习模型
    print("训练机器学习模型...")
    advisor.goal_predictor.train_models()
    
    # 示例分析
    print("\n" + "="*60)
    print("足球博彩建议系统演示")
    print("="*60)
    
    # 单场比赛详细分析
    advisor.print_detailed_analysis("manchester city", "arsenal")
    
    # 多场比赛分析
    sample_matches = [
        ("manchester city", "arsenal"),
        ("liverpool", "chelsea"),
        ("barcelona", "real madrid"),
        ("bayern münchen", "borussia dortmund")
    ]
    
    print(f"\n📋 多场比赛分析:")
    results_df = advisor.analyze_multiple_matches(sample_matches)
    if not results_df.empty:
        print(results_df.to_string(index=False))
    
    # 最佳投注建议
    print(f"\n🏆 今日最佳投注建议:")
    best_bets = advisor.get_best_bets_today(sample_matches, min_confidence=0.6)
    
    for i, bet in enumerate(best_bets, 1):
        risk_emoji = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🔴'}.get(bet['risk_level'], '⚪')
        print(f"{i}. {bet['match']} - {bet['suggestion']}")
        print(f"   预测比分: {bet['predicted_score']} | 置信度: {bet['confidence']:.2f} | 风险: {risk_emoji} {bet['risk_level']}")
        print()