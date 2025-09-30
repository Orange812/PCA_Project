#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于赔率增强的足球博彩分析系统
整合市场赔率数据，优化投注策略以实现正收益
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Tuple, Optional
from .elo_system import FootballBettingSystem
import warnings
warnings.filterwarnings('ignore')

class OddsEnhancedBettingSystem:
    """基于赔率增强的博彩系统"""
    
    def __init__(self, db_path: str = "football_betting.db"):
        self.db_path = db_path
        self.base_system = FootballBettingSystem(db_path=db_path)
        self.min_edge = 0.05  # 最小优势阈值 (5%)
        self.kelly_fraction = 0.25  # 凯利公式保守系数
        
    def odds_to_probability(self, odds: float) -> float:
        """将赔率转换为隐含概率"""
        if odds <= 1.0:
            return 0.0
        return 1.0 / odds
    
    def calculate_market_margin(self, odds_list: List[float]) -> float:
        """计算市场利润率"""
        total_prob = sum(self.odds_to_probability(odds) for odds in odds_list if odds > 1.0)
        return max(0, total_prob - 1.0)
    
    def find_value_bets(self, match_data: Dict) -> List[Dict]:
        """寻找价值投注机会"""
        value_bets = []
        
        # 获取我们的预测
        home_team = match_data['home_team_name']
        away_team = match_data['away_team_name']
        
        try:
            prediction = self.base_system.predict_match_goals(home_team, away_team)
            
            # 1. 胜负投注分析
            value_bets.extend(self._analyze_match_result_odds(match_data, prediction))
            
            # 2. 总进球数投注分析
            value_bets.extend(self._analyze_total_goals_odds(match_data, prediction))
            
            # 3. 双方进球投注分析
            value_bets.extend(self._analyze_btts_odds(match_data, prediction))
            
        except Exception as e:
            print(f"分析失败 {home_team} vs {away_team}: {e}")
        
        return value_bets
    
    def _analyze_match_result_odds(self, match_data: Dict, prediction: Dict) -> List[Dict]:
        """分析胜负投注赔率"""
        value_bets = []
        
        # 获取赔率
        home_odds = float(match_data.get('odds_ft_home_team_win', 0))
        draw_odds = float(match_data.get('odds_ft_draw', 0))
        away_odds = float(match_data.get('odds_ft_away_team_win', 0))
        
        if home_odds <= 1 or draw_odds <= 1 or away_odds <= 1:
            return value_bets
        
        # 获取我们的预测概率
        our_home_prob = prediction['home_win_probability']
        our_draw_prob = prediction['draw_probability'] 
        our_away_prob = prediction['away_win_probability']
        
        # 市场隐含概率
        market_home_prob = self.odds_to_probability(home_odds)
        market_draw_prob = self.odds_to_probability(draw_odds)
        market_away_prob = self.odds_to_probability(away_odds)
        
        # 计算优势
        home_edge = our_home_prob - market_home_prob
        draw_edge = our_draw_prob - market_draw_prob
        away_edge = our_away_prob - market_away_prob
        
        # 寻找价值投注
        if home_edge > self.min_edge:
            kelly_stake = self._calculate_kelly_stake(our_home_prob, home_odds)
            value_bets.append({
                'type': '胜负',
                'bet': '主胜',
                'odds': home_odds,
                'our_probability': our_home_prob,
                'market_probability': market_home_prob,
                'edge': home_edge,
                'kelly_stake': kelly_stake,
                'confidence': prediction['confidence']
            })
        
        if draw_edge > self.min_edge:
            kelly_stake = self._calculate_kelly_stake(our_draw_prob, draw_odds)
            value_bets.append({
                'type': '胜负',
                'bet': '平局',
                'odds': draw_odds,
                'our_probability': our_draw_prob,
                'market_probability': market_draw_prob,
                'edge': draw_edge,
                'kelly_stake': kelly_stake,
                'confidence': prediction['confidence']
            })
        
        if away_edge > self.min_edge:
            kelly_stake = self._calculate_kelly_stake(our_away_prob, away_odds)
            value_bets.append({
                'type': '胜负',
                'bet': '客胜',
                'odds': away_odds,
                'our_probability': our_away_prob,
                'market_probability': market_away_prob,
                'edge': away_edge,
                'kelly_stake': kelly_stake,
                'confidence': prediction['confidence']
            })
        
        return value_bets
    
    def _analyze_total_goals_odds(self, match_data: Dict, prediction: Dict) -> List[Dict]:
        """分析总进球数投注赔率"""
        value_bets = []
        
        predicted_total = prediction['total_goals_predicted']
        
        # 获取各个进球数档位的赔率
        over15_odds = float(match_data.get('odds_ft_over15', 0))
        over25_odds = float(match_data.get('odds_ft_over25', 0))
        over35_odds = float(match_data.get('odds_ft_over35', 0))
        over45_odds = float(match_data.get('odds_ft_over45', 0))
        
        # 基于我们的预测计算各档位概率
        our_over15_prob = self._calculate_over_probability(predicted_total, 1.5)
        our_over25_prob = self._calculate_over_probability(predicted_total, 2.5)
        our_over35_prob = self._calculate_over_probability(predicted_total, 3.5)
        our_over45_prob = self._calculate_over_probability(predicted_total, 4.5)
        
        # 分析每个档位
        goals_analysis = [
            (over15_odds, our_over15_prob, 'Over 1.5'),
            (over25_odds, our_over25_prob, 'Over 2.5'),
            (over35_odds, our_over35_prob, 'Over 3.5'),
            (over45_odds, our_over45_prob, 'Over 4.5')
        ]
        
        for odds, our_prob, bet_name in goals_analysis:
            if odds > 1:
                market_prob = self.odds_to_probability(odds)
                edge = our_prob - market_prob
                
                if edge > self.min_edge:
                    kelly_stake = self._calculate_kelly_stake(our_prob, odds)
                    value_bets.append({
                        'type': '总进球',
                        'bet': bet_name,
                        'odds': odds,
                        'our_probability': our_prob,
                        'market_probability': market_prob,
                        'edge': edge,
                        'kelly_stake': kelly_stake,
                        'confidence': prediction['confidence']
                    })
                
                # 同时分析Under投注
                under_prob = 1 - our_prob
                under_odds = 1 / (1 - market_prob) if market_prob < 0.99 else 1.01
                under_edge = under_prob - (1 - market_prob)
                
                if under_edge > self.min_edge and under_odds > 1:
                    kelly_stake = self._calculate_kelly_stake(under_prob, under_odds)
                    value_bets.append({
                        'type': '总进球',
                        'bet': bet_name.replace('Over', 'Under'),
                        'odds': under_odds,
                        'our_probability': under_prob,
                        'market_probability': 1 - market_prob,
                        'edge': under_edge,
                        'kelly_stake': kelly_stake,
                        'confidence': prediction['confidence']
                    })
        
        return value_bets
    
    def _analyze_btts_odds(self, match_data: Dict, prediction: Dict) -> List[Dict]:
        """分析双方进球投注赔率"""
        value_bets = []
        
        btts_yes_odds = float(match_data.get('odds_btts_yes', 0))
        btts_no_odds = float(match_data.get('odds_btts_no', 0))
        
        if btts_yes_odds <= 1 or btts_no_odds <= 1:
            return value_bets
        
        # 基于预测计算双方进球概率
        home_goals = prediction['home_goals_predicted']
        away_goals = prediction['away_goals_predicted']
        
        # 简化模型：假设进球数服从泊松分布
        home_no_goals_prob = np.exp(-home_goals)
        away_no_goals_prob = np.exp(-away_goals)
        
        our_btts_no_prob = home_no_goals_prob + away_no_goals_prob - home_no_goals_prob * away_no_goals_prob
        our_btts_yes_prob = 1 - our_btts_no_prob
        
        # 市场概率
        market_btts_yes_prob = self.odds_to_probability(btts_yes_odds)
        market_btts_no_prob = self.odds_to_probability(btts_no_odds)
        
        # 分析BTTS Yes
        btts_yes_edge = our_btts_yes_prob - market_btts_yes_prob
        if btts_yes_edge > self.min_edge:
            kelly_stake = self._calculate_kelly_stake(our_btts_yes_prob, btts_yes_odds)
            value_bets.append({
                'type': '双方进球',
                'bet': 'BTTS Yes',
                'odds': btts_yes_odds,
                'our_probability': our_btts_yes_prob,
                'market_probability': market_btts_yes_prob,
                'edge': btts_yes_edge,
                'kelly_stake': kelly_stake,
                'confidence': prediction['confidence']
            })
        
        # 分析BTTS No
        btts_no_edge = our_btts_no_prob - market_btts_no_prob
        if btts_no_edge > self.min_edge:
            kelly_stake = self._calculate_kelly_stake(our_btts_no_prob, btts_no_odds)
            value_bets.append({
                'type': '双方进球',
                'bet': 'BTTS No',
                'odds': btts_no_odds,
                'our_probability': our_btts_no_prob,
                'market_probability': market_btts_no_prob,
                'edge': btts_no_edge,
                'kelly_stake': kelly_stake,
                'confidence': prediction['confidence']
            })
        
        return value_bets
    
    def _calculate_over_probability(self, predicted_total: float, threshold: float) -> float:
        """计算超过某个进球数的概率（基于正态分布近似）"""
        # 假设进球数的标准差约为预测值的平方根
        std = np.sqrt(predicted_total)
        
        # 使用正态分布的累积分布函数
        from scipy.stats import norm
        try:
            prob = 1 - norm.cdf(threshold, predicted_total, std)
            return max(0.01, min(0.99, prob))
        except:
            # 如果scipy不可用，使用简化计算
            if predicted_total > threshold:
                return 0.6 + (predicted_total - threshold) * 0.1
            else:
                return 0.4 - (threshold - predicted_total) * 0.1
    
    def _calculate_kelly_stake(self, win_probability: float, odds: float) -> float:
        """计算凯利公式投注比例"""
        if odds <= 1 or win_probability <= 0:
            return 0
        
        # 凯利公式: f = (bp - q) / b
        # 其中 b = odds - 1, p = win_probability, q = 1 - p
        b = odds - 1
        p = win_probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # 应用保守系数并限制最大投注比例
        conservative_kelly = kelly_fraction * self.kelly_fraction
        return max(0, min(0.1, conservative_kelly))  # 最大10%的资金
    
    def backtest_strategy(self, start_date: str = "2023-01-01", end_date: str = "2024-01-01") -> Dict:
        """回测投注策略"""
        print(f"开始回测策略 ({start_date} 到 {end_date})")
        
        # 获取历史比赛数据
        conn = sqlite3.connect(self.db_path)
        
        # 从原始CSV文件获取包含赔率的数据
        all_matches = []
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
                    df['league'] = f"{country}-{league}"
                    df['season'] = season
                    all_matches.append(df)
                except FileNotFoundError:
                    continue
        
        if not all_matches:
            return {"error": "没有找到历史数据"}
        
        combined_df = pd.concat(all_matches, ignore_index=True)
        
        # 过滤日期范围（如果有日期字段）
        if 'date_GMT' in combined_df.columns:
            combined_df['date_GMT'] = pd.to_datetime(combined_df['date_GMT'], errors='coerce')
            mask = (combined_df['date_GMT'] >= start_date) & (combined_df['date_GMT'] <= end_date)
            combined_df = combined_df[mask]
        
        # 回测结果
        total_bets = 0
        winning_bets = 0
        total_stake = 0
        total_return = 0
        bet_details = []
        
        print(f"分析 {len(combined_df)} 场比赛...")
        
        for idx, match in combined_df.iterrows():
            if idx % 100 == 0:
                print(f"进度: {idx}/{len(combined_df)}")
            
            try:
                # 寻找价值投注
                value_bets = self.find_value_bets(match.to_dict())
                
                for bet in value_bets:
                    if bet['confidence'] < 0.3:  # 跳过低置信度的投注
                        continue
                    
                    total_bets += 1
                    stake = bet['kelly_stake'] * 100  # 假设总资金100单位
                    total_stake += stake
                    
                    # 判断投注结果
                    won = self._check_bet_result(match, bet)
                    
                    if won:
                        winning_bets += 1
                        return_amount = stake * bet['odds']
                        total_return += return_amount
                        profit = return_amount - stake
                    else:
                        profit = -stake
                    
                    bet_details.append({
                        'match': f"{match['home_team_name']} vs {match['away_team_name']}",
                        'bet_type': bet['type'],
                        'bet': bet['bet'],
                        'odds': bet['odds'],
                        'stake': stake,
                        'won': won,
                        'profit': profit,
                        'edge': bet['edge']
                    })
            
            except Exception as e:
                continue
        
        # 计算回测结果
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        total_profit = total_return - total_stake
        roi = (total_profit / total_stake * 100) if total_stake > 0 else 0
        
        results = {
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'win_rate': win_rate,
            'total_stake': total_stake,
            'total_return': total_return,
            'total_profit': total_profit,
            'roi': roi,
            'average_odds': np.mean([bet['odds'] for bet in bet_details]) if bet_details else 0,
            'bet_details': bet_details[:50]  # 只返回前50个详细记录
        }
        
        conn.close()
        return results
    
    def _check_bet_result(self, match: pd.Series, bet: Dict) -> bool:
        """检查投注结果是否获胜"""
        home_goals = match['home_team_goal_count']
        away_goals = match['away_team_goal_count']
        total_goals = home_goals + away_goals
        
        bet_type = bet['type']
        bet_name = bet['bet']
        
        if bet_type == '胜负':
            if bet_name == '主胜':
                return home_goals > away_goals
            elif bet_name == '平局':
                return home_goals == away_goals
            elif bet_name == '客胜':
                return home_goals < away_goals
        
        elif bet_type == '总进球':
            if 'Over' in bet_name:
                threshold = float(bet_name.split()[-1])
                return total_goals > threshold
            elif 'Under' in bet_name:
                threshold = float(bet_name.split()[-1])
                return total_goals < threshold
        
        elif bet_type == '双方进球':
            if bet_name == 'BTTS Yes':
                return home_goals > 0 and away_goals > 0
            elif bet_name == 'BTTS No':
                return home_goals == 0 or away_goals == 0
        
        return False

# 使用示例
if __name__ == "__main__":
    # 创建增强系统
    enhanced_system = OddsEnhancedBettingSystem()
    
    # 初始化基础系统
    print("初始化基础预测系统...")
    enhanced_system.base_system.run_full_analysis()
    
    # 回测策略
    print("\n开始回测投注策略...")
    backtest_results = enhanced_system.backtest_strategy()
    
    if 'error' not in backtest_results:
        print(f"\n=== 回测结果 ===")
        print(f"总投注次数: {backtest_results['total_bets']}")
        print(f"获胜次数: {backtest_results['winning_bets']}")
        print(f"胜率: {backtest_results['win_rate']:.2%}")
        print(f"总投注金额: {backtest_results['total_stake']:.2f}")
        print(f"总回报: {backtest_results['total_return']:.2f}")
        print(f"净利润: {backtest_results['total_profit']:.2f}")
        print(f"投资回报率: {backtest_results['roi']:.2f}%")
        print(f"平均赔率: {backtest_results['average_odds']:.2f}")
        
        # 显示部分投注详情
        print(f"\n=== 投注详情示例 ===")
        for i, bet in enumerate(backtest_results['bet_details'][:10]):
            result = "✅ 获胜" if bet['won'] else "❌ 失败"
            print(f"{i+1}. {bet['match']} - {bet['bet']} @{bet['odds']:.2f} - {result} (利润: {bet['profit']:.2f})")
    else:
        print(f"回测失败: {backtest_results['error']}")