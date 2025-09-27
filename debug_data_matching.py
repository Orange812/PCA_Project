# debug_data_matching.py
# 调试数据匹配问题

import pandas as pd
import os

def debug_data_matching():
    print("调试数据匹配问题...")
    
    # 检查ELO数据
    elo_df = pd.read_csv("xg_elo_scores.csv")
    print(f"ELO数据: {len(elo_df)} 条记录")
    print("ELO数据联赛分布:")
    print(elo_df['league'].value_counts())
    print("\nELO数据赛季分布:")
    print(elo_df['season'].value_counts())
    
    # 检查一个具体的比赛文件
    match_file = "data/england-premier-league-matches-2023-to-2024-stats.csv"
    if os.path.exists(match_file):
        match_df = pd.read_csv(match_file)
        print(f"\n英超2023-24赛季比赛数据: {len(match_df)} 场比赛")
        print("球队名称样例:")
        print(match_df['home_team_name'].unique()[:10])
        
        # 检查ELO数据中对应的球队
        elo_epl_2023 = elo_df[
            (elo_df['league'] == 'premier-league') & 
            (elo_df['season'] == '2023-to-2024')
        ]
        print(f"\nELO数据中英超2023-24: {len(elo_epl_2023)} 条记录")
        if len(elo_epl_2023) > 0:
            print("ELO数据中的球队名称:")
            print(elo_epl_2023['team_name'].unique())
        
        # 检查球队排名文件
        team_file = "data/england-premier-league-teams-2023-to-2024-stats.csv"
        if os.path.exists(team_file):
            team_df = pd.read_csv(team_file)
            print(f"\n英超2023-24球队数据: {len(team_df)} 支球队")
            print("排名前10的球队:")
            top_teams = team_df.sort_values('league_position').head(10)
            print(top_teams[['team_name', 'common_name', 'league_position']])
            
            print("\n中下游球队（排名7-15）:")
            middle_teams = team_df.sort_values('league_position').iloc[6:15]
            print(middle_teams[['team_name', 'common_name', 'league_position']])

if __name__ == "__main__":
    debug_data_matching()