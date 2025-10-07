

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# 从我们之前创建的脚本中导入数据加载和特征工程函数
from feature_engineering_corners import load_all_matches_from_csvs, calculate_corner_features

def train_and_evaluate_model():
    """
    完整的模型训练和评估流程。
    """
    # 1. 加载和准备数据
    print("正在加载和处理数据...")
    data_directory = 'data'
    matches_df = load_all_matches_from_csvs(data_directory)
    matches_df.dropna(subset=['home_team_corner_count', 'away_team_corner_count'], inplace=True)
    matches_with_features = calculate_corner_features(matches_df)
    
    # 清理用于建模的最终数据集
    final_df = matches_with_features.dropna(subset=[
        'home_avg_corners_for', 'home_avg_corners_against',
        'away_avg_corners_for', 'away_avg_corners_against',
        'total_corner_count'
    ])
    print(f"数据准备就绪，共 {len(final_df)} 条记录用于建模。")

    # 2. 定义特征和目标
    features = [
        'home_avg_corners_for', 'home_avg_corners_against',
        'away_avg_corners_for', 'away_avg_corners_against'
    ]
    target = 'total_corner_count'

    X = final_df[features]
    y = final_df[target]

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"数据集划分为 {len(X_train)} 条训练记录和 {len(X_test)} 条测试记录。")

    # 4. 训练模型
    print("开始训练梯度提升模型...")
    # 使用一些合理的默认参数以加快训练速度
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    gbr.fit(X_train, y_train)
    print("模型训练完成。")

    # 5. 评估模型
    print("正在评估模型性能...")
    predictions = gbr.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print("--- 模型评估结果 ---")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print("这意味着模型对比赛总角球数的预测平均误差约为 {:.2f} 个角球。".format(mae))
    
    # 6. 显示一些预测样本
    result_df = X_test.copy()
    result_df['actual_corners'] = y_test
    result_df['predicted_corners'] = predictions
    print("\n--- 预测样本 ---")
    print(result_df.head())

if __name__ == '__main__':
    train_and_evaluate_model()

