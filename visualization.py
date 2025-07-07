# visualization.py
# 描述: 包含用于数据可视化的函数。

import os
import matplotlib.pyplot as plt


def visualize_team_evolution_by_league_static(team_positions_df, seasons_order, output_dir):
    """按联赛可视化球队演变"""
    os.makedirs(output_dir, exist_ok=True)

    for league in team_positions_df['league'].unique():
        league_df = team_positions_df[team_positions_df['league'] == league].copy()

        # 筛选出在所有指定赛季都存在的球队
        team_counts = league_df.groupby("team_name")['season'].nunique()
        valid_teams = team_counts[team_counts == len(seasons_order)].index.unique()

        # 如果没有球队贯穿所有赛季，则使用所有球队
        if valid_teams.empty:
            valid_teams = league_df['team_name'].unique()
            print(f"在联赛 {league} 中未找到贯穿所有赛季的球队，将绘制所有球队。")

        valid_df = league_df[league_df['team_name'].isin(valid_teams)].sort_values(['team_name', 'season'])

        plt.figure(figsize=(12, 10))
        plt.title(f"{league.replace('-', ' ').title()} - Team Evolution (Offense-Defense Space)", fontsize=16)
        plt.xlabel("PC1 (Attacking Performance)", fontsize=12)
        plt.ylabel("PC2 (Defensive Performance)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        for team in valid_df['team_name'].unique():
            sub = valid_df[valid_df['team_name'] == team]
            plt.plot(sub['PC1'], sub['PC2'], marker='o', linestyle='-', label=team.title(), alpha=0.8)
            for _, row in sub.iterrows():
                # 提取年份的后两位
                year = row['season'].split('-')[-1]
                plt.text(row['PC1'], row['PC2'], f"'{year[2:]}", fontsize=8, ha='right', va='bottom')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # 调整布局为图例留出空间
        filename = os.path.join(output_dir, f"{league}_evolution.png")
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"已保存 {league} 演变图 -> {filename}")