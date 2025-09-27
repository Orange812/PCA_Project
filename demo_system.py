#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
足球博彩预测分析系统演示
展示核心功能和预测结果
"""

from football_betting_system import FootballBettingSystem
import pandas as pd

def main():
    print("="*80)
    print("🏆 足球博彩预测分析系统演示")
    print("="*80)
    
    # 初始化系统
    print("\n📊 正在初始化系统...")
    system = FootballBettingSystem()
    
    # 运行完整分析
    optimal_recent_matches = system.run_full_analysis()
    
    print(f"\n✅ 系统初始化完成!")
    print(f"   📈 主客场优势系数: {system.home_advantage:.3f}")
    print(f"   🎯 最优历史比赛场次数: {optimal_recent_matches}")
    print(f"   💾 数据库: {system.db_path}")
    
    # 演示预测功能
    print(f"\n" + "="*80)
    print("🔮 比赛预测演示")
    print("="*80)
    
    # 示例比赛
    demo_matches = [
        ("manchester city", "arsenal", "英超豪门对决"),
        ("liverpool", "chelsea", "英超强强对话"),
        ("real madrid", "fc barcelona", "西甲国家德比"),
        ("bayern münchen", "borussia dortmund", "德甲经典对决"),
        ("juventus", "ac milan", "意甲传统强队"),
        ("psg", "olympique marseille", "法甲焦点战")
    ]
    
    results = []
    
    for home_team, away_team, description in demo_matches:
        try:
            print(f"\n🏟️  {description}: {home_team.title()} vs {away_team.title()}")
            
            prediction = system.predict_match_goals(home_team, away_team, optimal_recent_matches)
            
            # 格式化输出
            home_goals = prediction['home_goals_predicted']
            away_goals = prediction['away_goals_predicted']
            total_goals = prediction['total_goals_predicted']
            confidence = prediction['confidence']
            
            print(f"   📊 ELO分数: {prediction['home_elo']:.0f} vs {prediction['away_elo']:.0f}")
            print(f"   ⚽ 预测比分: {home_goals:.1f} - {away_goals:.1f}")
            print(f"   🎯 总进球数: {total_goals:.1f}")
            print(f"   📈 胜负概率: 主胜 {prediction['home_win_probability']:.1%} | 平局 {prediction['draw_probability']:.1%} | 客胜 {prediction['away_win_probability']:.1%}")
            print(f"   🔒 置信度: {confidence:.2f}")
            
            # 生成简单的博彩建议
            suggestions = generate_simple_betting_advice(prediction)
            if suggestions:
                print(f"   💡 建议: {suggestions}")
            
            # 保存结果用于汇总
            results.append({
                'match': f"{home_team.title()} vs {away_team.title()}",
                'description': description,
                'predicted_score': f"{home_goals:.1f} - {away_goals:.1f}",
                'total_goals': total_goals,
                'home_win_prob': prediction['home_win_probability'],
                'confidence': confidence,
                'suggestion': suggestions
            })
            
        except Exception as e:
            print(f"   ❌ 预测失败: {e}")
            continue
    
    # 汇总结果
    if results:
        print(f"\n" + "="*80)
        print("📋 预测结果汇总")
        print("="*80)
        
        df = pd.DataFrame(results)
        
        # 按置信度排序
        df = df.sort_values('confidence', ascending=False)
        
        print(f"\n🏆 最可靠的预测 (按置信度排序):")
        for i, row in df.head(3).iterrows():
            print(f"{i+1}. {row['match']} - {row['predicted_score']}")
            print(f"   置信度: {row['confidence']:.2f} | 建议: {row['suggestion']}")
        
        print(f"\n⚽ 进球数预测统计:")
        avg_total_goals = df['total_goals'].mean()
        high_scoring = len(df[df['total_goals'] >= 3.0])
        low_scoring = len(df[df['total_goals'] <= 2.0])
        
        print(f"   平均总进球数: {avg_total_goals:.1f}")
        print(f"   高进球比赛 (≥3球): {high_scoring}/{len(df)}")
        print(f"   低进球比赛 (≤2球): {low_scoring}/{len(df)}")
        
        print(f"\n📊 主客场优势分析:")
        home_wins = len(df[df['home_win_prob'] > 0.5])
        print(f"   主队优势明显的比赛: {home_wins}/{len(df)}")
        print(f"   平均主队胜率: {df['home_win_prob'].mean():.1%}")
    
    print(f"\n" + "="*80)
    print("🎯 系统特色功能总结")
    print("="*80)
    print("✅ ELO评分系统 - 量化球队实力")
    print("✅ 主客场优势计算 - 基于历史数据统计")
    print("✅ ELO加权进球预测 - 考虑对手实力的进球分析")
    print("✅ 数据清洗 - 自动剔除异常比赛(净胜球>4)")
    print("✅ 最优参数寻找 - 交叉验证确定最佳历史比赛场次")
    print("✅ 置信度评估 - 预测可靠性量化")
    print("✅ 博彩建议生成 - 基于预测结果的投注建议")
    
    print(f"\n🚀 系统演示完成! 数据已保存至 {system.db_path}")

def generate_simple_betting_advice(prediction):
    """生成简单的博彩建议"""
    total_goals = prediction['total_goals_predicted']
    confidence = prediction['confidence']
    home_prob = prediction['home_win_probability']
    
    if confidence < 0.3:
        return "置信度较低，建议谨慎投注"
    
    suggestions = []
    
    # 总进球数建议
    if total_goals >= 3.5:
        suggestions.append("大球(Over 3.5)")
    elif total_goals >= 2.5:
        suggestions.append("大球(Over 2.5)")
    elif total_goals <= 1.5:
        suggestions.append("小球(Under 1.5)")
    else:
        suggestions.append("小球(Under 2.5)")
    
    # 胜负建议
    if home_prob > 0.6:
        suggestions.append("主胜")
    elif home_prob < 0.3:
        suggestions.append("客胜")
    else:
        suggestions.append("胜负难料")
    
    return " | ".join(suggestions)

if __name__ == "__main__":
    main()