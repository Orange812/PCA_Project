# -*- coding: utf-8 -*-
"""
@File: map_team_names.py
@Description:
该脚本用于自动匹配两个来源的球队名称，并生成一个映射字典。
流程如下：
1. 检查必要的库 a`thefuzz` 是否存在。
2. 加载之前诊断出的两个不匹配队名列表文件。
3. 使用模糊字符串匹配算法，为简称（比赛数据中）在全称列表（球队数据中）里寻找最佳匹配。
4. 将置信度高的匹配存入一个字典，置信度低的放入一个待办列表。
5. 打印结果供用户审查。
@Author: Gemini
@Date: 2025-08-23
"""

import sys

# 步骤1: 检查依赖库
try:
    from thefuzz import process as fuzz_process
except ImportError:
    print("错误：缺少必要的库 `thefuzz` 和 `python-Levenshtein`。", file=sys.stderr)
    print("请在您的终端中运行以下命令来安装它们:", file=sys.stderr)
    print("pip install thefuzz python-Levenshtein", file=sys.stderr)
    sys.exit(1)

def load_names_from_file(filename: str) -> list[str]:
    """从文件中加载队名列表，每行一个。"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # 使用strip()移除每行末尾的换行符
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"错误：找不到文件 '{filename}'。请确保该文件存在于当前目录。", file=sys.stderr)
        return []

def main():
    """
    主函数，执行名称匹配流程。
    """
    print("--- 开始执行球队名称自动匹配任务 ---")

    # 步骤2: 加载不匹配的队名列表
    matches_names_file = 'unmatched_teams_in_matches_data.txt'
    teams_names_file = 'unmatched_teams_in_teams_data.txt'

    matches_names = load_names_from_file(matches_names_file)
    teams_names = load_names_from_file(teams_names_file)

    if not matches_names or not teams_names:
        print("一个或两个队名列表文件为空或未找到，任务中止。")
        return

    print(f"成功加载 {len(matches_names)} 个比赛队名和 {len(teams_names)} 个球队统计队名。")

    # 步骤3 & 4: 执行模糊匹配并分类
    high_confidence_map = {}
    needs_manual_review = []
    MATCH_THRESHOLD = 85  # 定义置信度阈值

    print(f"\n正在进行模糊匹配（置信度阈值 > {MATCH_THRESHOLD}）...")

    for name in matches_names:
        # fuzz_process.extractOne会找到最佳匹配项及其分数
        best_match, score = fuzz_process.extractOne(name, teams_names)
        
        if score > MATCH_THRESHOLD:
            # 如果分数高于阈值，我们认为这是一个高置信度匹配
            high_confidence_map[name] = best_match
        else:
            # 否则，放入待审查列表
            needs_manual_review.append(name)
    
    print("匹配完成。")

    # 步骤5: 打印结果供审查
    print("\n---!!! 审查以下自动匹配结果 !!!---")
    print(f"\n成功找到 {len(high_confidence_map)} 个高置信度匹配：")
    # 为了方便查看，格式化打印字典
    for key, value in high_confidence_map.items():
        print(f"  '{key}'  ->  '{value}'")

    print(f"\n有 {len(needs_manual_review)} 个队名需要手动审查或网络搜索：")
    print(needs_manual_review)
    
    print("\n--- 任务完成 ---")
    print("请您审查以上高置信度匹配。如果无误，我们下一步将处理需要手动审查的列表。")

if __name__ == '__main__':
    main()
