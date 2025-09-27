# -*- coding: utf-8 -*-
"""
@File: finalize_mapping.py
@Description:
该脚本结合自动匹配和用户提供的手动匹配，生成最终的队名映射文件。
流程如下：
1. 重新运行模糊匹配，生成高置信度匹配字典。
2. 定义用户手动提供的15个队名的映射关系。
3. 为用户提供的简称，从全称列表中找到最相似的官方全称。
4. 合并高置信度字典和手动映射字典。
5. 将最终的完整映射字典保存为 team_name_map.json。
@Author: Gemini
@Date: 2025-08-23
"""

import json
import sys

try:
    from thefuzz import process as fuzz_process
except ImportError:
    print("错误：缺少必要的库 `thefuzz`。请重新运行 `pip install thefuzz python-Levenshtein`", file=sys.stderr)
    sys.exit(1)

def load_names_from_file(filename: str) -> list[str]:
    """从文件中加载队名列表。"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"错误：找不到文件 '{filename}'。", file=sys.stderr)
        return []

def main():
    """
    主函数，执行映射生成和保存流程。
    """
    print("--- 开始生成最终的球队名称映射文件 ---")

    # 1. 加载队名列表
    matches_names_file = 'unmatched_teams_in_matches_data.txt'
    teams_names_file = 'unmatched_teams_in_teams_data.txt'
    matches_names_all = load_names_from_file(matches_names_file)
    teams_names_all = load_names_from_file(teams_names_file)

    if not matches_names_all or not teams_names_all:
        print("队名列表文件为空或未找到，任务中止。")
        return

    # 2. 重新生成高置信度匹配
    high_confidence_map = {}
    MATCH_THRESHOLD = 85
    unmatched_after_auto = []

    for name in matches_names_all:
        best_match, score = fuzz_process.extractOne(name, teams_names_all)
        if score > MATCH_THRESHOLD:
            high_confidence_map[name] = best_match
        else:
            unmatched_after_auto.append(name)
    
    print(f"自动匹配完成，生成了 {len(high_confidence_map)} 个高置信度映射。")

    # 3. 处理用户提供的手动映射
    # 这是您提供的列表
    user_provided_list = [
        'AGF', 'AaB', 'Auxerre', "Borussia M'gladbach", 'Concarneau', 'Darmstadt 98', 
        'Hearts', 'Inter Milan', 'PSG', 'RWDM', 'Rennes', 'SPAL', 'Sint-Truiden', 
        'St. Johnstone', 'SønderjyskE'
    ]
    
    manual_map = {}
    print("\n正在根据您的手动列表，处理剩余的名称...")
    for name in user_provided_list:
        # 即使是手动列表，我们依然从全称列表中找到最精确的匹配项，以保证名称的准确性
        best_match, score = fuzz_process.extractOne(name, teams_names_all)
        if best_match:
            manual_map[name] = best_match
            print(f"  映射 '{name}' -> '{best_match}'")

    # 4. 合并两个字典
    final_map = {**high_confidence_map, **manual_map}
    print(f"\n合并完成。最终映射包含 {len(final_map)} 个条目。")

    # 5. 保存到JSON文件
    map_filename = 'team_name_map.json'
    with open(map_filename, 'w', encoding='utf-8') as f:
        json.dump(final_map, f, indent=4, ensure_ascii=False)
    
    print(f"\n成功将最终的映射字典保存到: {map_filename}")
    print("下一步，我们将修改主脚本来使用这个文件。")

if __name__ == '__main__':
    main()
