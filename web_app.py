
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from scipy.stats import poisson

app = Flask(__name__)

def predict_corner_lambdas(features):
    """
    占位符模型：根据输入特征预测主客队的预期角球数 (lambda)。
    注意：这是一个简化的占位符，用于演示端到端的功能。
    在实际应用中，这里应该加载一个预先训练好的机器学习模型（如 .pkl 文件）并调用其 predict 方法。
    """
    # 为不同特征设定一个合理的、基于经验的权重
    weights = {
        'ppg': 0.1,
        'avg_shots': 0.2,
        'avg_shots_on_target': 0.15,
        'xg': 1.5,
        'avg_possession': 0.05,
        'corners_for': 0.3,
        'corners_against': -0.1 
    }

    # 计算主队 lambda
    home_lambda = (
        features['home_ppg'] * weights['ppg'] +
        features['home_avg_shots'] * weights['avg_shots'] +
        features['home_avg_shots_on_target'] * weights['avg_shots_on_target'] +
        features['home_xg'] * weights['xg'] +
        features['home_avg_possession'] * weights['avg_possession'] +
        features['home_corners_for'] * weights['corners_for'] +
        features['away_corners_against'] * weights['corners_for'] # 客队的防守送角也会影响主队角球
    )

    # 计算客队 lambda
    away_lambda = (
        features['away_ppg'] * weights['ppg'] +
        features['away_avg_shots'] * weights['avg_shots'] +
        features['away_avg_shots_on_target'] * weights['avg_shots_on_target'] +
        features['away_xg'] * weights['xg'] +
        features['away_avg_possession'] * weights['avg_possession'] +
        features['away_corners_for'] * weights['corners_for'] +
        features['home_corners_against'] * weights['corners_for'] # 主队的防守送角也会影响客队角球
    )
    
    # 确保 lambda 值为正
    return max(0.5, home_lambda), max(0.5, away_lambda)

@app.route('/')
def index():
    return redirect(url_for('predict'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # 1. 从表单获取所有特征数据并转换为浮点数
        try:
            features = {key: float(value) for key, value in request.form.items()}
        except (ValueError, TypeError):
            return "Invalid input format. Please ensure all fields are numbers.", 400

        # 2. 调用模型获取 lambda 值
        home_lambda, away_lambda = predict_corner_lambdas(features)
        
        total_corners_prediction = home_lambda + away_lambda

        # 3. 计算泊松分布矩阵 (最高计算到15个角球)
        max_corners = 16
        corner_matrix = np.zeros((max_corners, max_corners))
        for home_goals in range(max_corners):
            for away_goals in range(max_corners):
                home_prob = poisson.pmf(home_goals, home_lambda)
                away_prob = poisson.pmf(away_goals, away_lambda)
                corner_matrix[home_goals, away_goals] = home_prob * away_prob

        # 4. 计算总角球数概率分布
        total_corners_dist = {}
        for total in range(max_corners * 2 -1):
            prob = 0
            for i in range(max_corners):
                for j in range(max_corners):
                    if i + j == total:
                        prob += corner_matrix[i, j]
            if prob > 0.001: # 只显示有一定可能性的结果
                total_corners_dist[total] = prob
        
        # 5. 渲染结果页面
        return render_template(
            'corner_predict_result.html',
            home_lambda=f"{home_lambda:.2f}",
            away_lambda=f"{away_lambda:.2f}",
            total_prediction=f"{total_corners_prediction:.2f}",
            matrix=corner_matrix,
            total_dist=total_corners_dist,
            max_corners_range=range(max_corners)
        )

    # 如果是 GET 请求，显示输入表单
    return render_template('corner_predict_form.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
