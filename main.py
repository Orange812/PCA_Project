# main.py
# 描述: 项目的主执行文件，负责协调所有模块完成整个分析流程。

import sklearn
import config
from data_loader import load_all_league_data
from modeling import compute_adaboost_weights, random_search_hyperparameters, adam_optimize_positions
from visualization import visualize_team_evolution_by_league_static


def main():
    """主函数，执行整个分析流程"""
    print(f"Scikit-learn 版本: {sklearn.__version__}")

    # 1. 加载和预处理数据
    print("\n" + "=" * 20 + " 1. 加载与预处理数据 " + "=" * 20)
    all_team_positions, all_match_positions = load_all_league_data(
        base_path=config.BASE_PATH,
        leagues=config.LEAGUES,
        seasons=config.SEASONS
    )

    if all_team_positions.empty or all_match_positions.empty:
        print("数据加载失败或数据为空，程序终止。")
        return

    # 2. 计算Adaboost权重
    print("\n" + "=" * 20 + " 2. 计算Adaboost权重 " + "=" * 20)
    adaboost_weights = compute_adaboost_weights(all_team_positions)
    print(f"优化后的Adaboost权重: {adaboost_weights}")

    # 3. 随机搜索超参数
    print("\n" + "=" * 20 + " 3. 随机搜索超参数 " + "=" * 20)
    best_params = random_search_hyperparameters(
        all_team_positions, all_match_positions, w=adaboost_weights, n_iter=10
    )

    # 4. 使用最佳参数进行最终训练
    print("\n" + "=" * 20 + " 4. 最终模型训练 " + "=" * 20)
    final_losses, final_team_positions, final_rank_scale, final_w = adam_optimize_positions(
        all_team_positions.copy(),
        all_match_positions.copy(),
        w=adaboost_weights,
        **best_params
    )

    if not final_losses:
        print("最终训练未能产生结果，程序终止。")
        return

    print(f"最终损失: {final_losses[-1]:.4f}")
    print(f"最终Rank Scale: {final_rank_scale:.4f}")
    print(f"最终权重: {final_w}")

    # 保存最终结果到CSV
    final_team_positions.to_csv(config.FINAL_CSV_PATH, index=False)
    print(f"最终球队位置数据已保存到 {config.FINAL_CSV_PATH}")

    # 5. 可视化结果
    print("\n" + "=" * 20 + " 5. 生成可视化图表 " + "=" * 20)
    visualize_team_evolution_by_league_static(
        final_team_positions,
        seasons_order=config.SEASONS,
        output_dir=config.OUTPUT_DIR
    )

    print("\n" + "=" * 20 + " 分析流程全部完成！ " + "=" * 20)


if __name__ == "__main__":
    main()