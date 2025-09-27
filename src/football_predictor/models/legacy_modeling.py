# modeling.py
# 描述: 包含Adaboost权重计算、TensorFlow损失函数和优化器，以及超参数搜索。

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform
from feature_engineering import initialize_elo_scores


# ====================== 使用Adaboost优化权重 ======================
def compute_adaboost_weights(team_positions_df):
    """使用Adaboost计算防守比率的最佳权重，并通过交叉验证优化参数"""
    ratios = team_positions_df[['ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5', 'ratio6']].replace([np.inf, -np.inf],
                                                                                                     np.nan).fillna(0)
    print("特征相关性矩阵:\n", ratios.corr())

    scaler = StandardScaler()
    ratios_scaled = scaler.fit_transform(ratios)
    labels = pd.qcut(team_positions_df['average_goals_conceded'], q=2, labels=[0, 1], duplicates='drop').fillna(
        0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(ratios_scaled, labels, test_size=0.2, random_state=42)

    ada = AdaBoostClassifier(algorithm='SAMME', random_state=42)
    param_grid = {
        'estimator': [DecisionTreeClassifier(max_depth=5), DecisionTreeClassifier(max_depth=10)],
        'n_estimators': [50, 100, 150, 200, 250, 500, 1000, 1500],
        'learning_rate': [0.01, 0.05, 0.1, 0.25, 0.5]
    }

    grid_search = GridSearchCV(ada, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("最佳参数:", best_params)

    adaboost = AdaBoostClassifier(
        estimator=best_params['estimator'], n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'], algorithm='SAMME', random_state=42
    )
    adaboost.fit(X_train, y_train)

    y_pred = adaboost.predict(X_test)
    print("测试集准确率:", accuracy_score(y_test, y_pred))
    print("测试集F1分数:", f1_score(y_test, y_pred, average='weighted'))
    print("测试集ROC-AUC:", roc_auc_score(y_test, adaboost.predict_proba(X_test)[:, 1]))
    print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))

    feature_importances = adaboost.feature_importances_
    weights = feature_importances / np.sum(feature_importances)
    print(f"计算得到的权重: {weights}")

    return weights.astype(np.float32)


# ====================== TensorFlow 优化逻辑 ======================
def compute_total_loss(positions, match_home_idx, match_away_idx, match_PC1, match_PC2, points_per_game, rank_scale,
                       ratios, w, normalized_defense_score, lambda_defense, lambda_supervision, lambda_reg, elo_scores,
                       home_away_diff):
    """计算总损失，考虑主客场表现差异比值"""
    epsilon = 1e-8
    if tf.shape(match_home_idx)[0] == 0:
        return tf.constant(0.0, dtype=tf.float32)

    home_pos = tf.gather(positions, match_home_idx)
    away_pos = tf.gather(positions, match_away_idx)
    match_points = tf.stack([match_PC1, match_PC2], axis=1)
    dist_home = tf.norm(home_pos - match_points + epsilon, axis=1)
    dist_away = tf.norm(away_pos - match_points + epsilon, axis=1)

    all_distances = tf.concat([dist_home, dist_away], axis=0)
    dist_range = tf.reduce_max(all_distances) - tf.reduce_min(all_distances) + epsilon
    dist_home_norm = (dist_home - tf.reduce_min(all_distances)) / dist_range
    dist_away_norm = (dist_away - tf.reduce_min(all_distances)) / dist_range

    home_elo = tf.gather(elo_scores, match_home_idx)
    away_elo = tf.gather(elo_scores, match_away_idx)
    home_diff = tf.gather(home_away_diff, match_home_idx)
    away_diff = tf.gather(home_away_diff, match_away_idx)

    elo_diff = tf.abs(home_elo - away_elo)
    weight_home = 1.0 / (1.0 + elo_diff * rank_scale + epsilon)
    weight_away = weight_home * away_diff
    total_weight = weight_home + weight_away
    weight_home = weight_home / (total_weight + epsilon)
    weight_away = weight_away / (total_weight + epsilon)

    match_loss = tf.reduce_mean(weight_home * dist_home_norm + weight_away * dist_away_norm)

    w_abs = tf.abs(w)
    defense_target = -tf.reduce_sum(w_abs * ratios, axis=1)
    defense_loss = tf.reduce_mean(tf.square(positions[:, 1] - defense_target))
    supervision_loss = tf.reduce_mean(tf.square(defense_target - normalized_defense_score))
    regularization_loss = lambda_reg * tf.reduce_sum(tf.square(w))

    total_loss = match_loss + lambda_defense * defense_loss + lambda_supervision * supervision_loss + regularization_loss
    return tf.where(tf.math.is_nan(total_loss) | tf.math.is_inf(total_loss), 0.0, total_loss)


def adam_optimize_positions(team_positions_df, match_positions_df, initial_lr=0.0005, decay_steps=200000,
                            decay_rate=0.9, clipnorm=0.5, iterations=30000, verbose_interval=1000, random_seed=42,
                            lambda_defense=0.1, lambda_supervision=0.1, lambda_reg=0.01, patience=100, w=None):
    """使用Adam优化球队位置，考虑主客场差异"""
    team_seasons = team_positions_df['team_season'].unique()
    team_season_to_idx = {t: i for i, t in enumerate(team_seasons)}

    elo_scores_dict = initialize_elo_scores(team_positions_df)
    team_names = [t.split('_')[1] for t in team_seasons]
    elo_scores = tf.convert_to_tensor([elo_scores_dict.get(name, 1500) for name in team_names], dtype=tf.float32)

    home_away_diff = tf.convert_to_tensor(team_positions_df['home_away_diff'].values, dtype=tf.float32)

    team_positions_df = team_positions_df.set_index('team_season')
    init_positions = team_positions_df[['PC1', 'PC2']].values
    ratios_vals = team_positions_df[['ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5', 'ratio6']].values
    scaler_pos = StandardScaler()
    scaler_rat = StandardScaler()
    positions_scaled = scaler_pos.fit_transform(init_positions)
    ratios_scaled = scaler_rat.fit_transform(ratios_vals)

    positions = tf.Variable(positions_scaled, dtype=tf.float32)
    ratios = tf.constant(ratios_scaled, dtype=tf.float32)
    normalized_defense_score = tf.constant(team_positions_df['normalized_defense_score'].values, dtype=tf.float32)
    points_per_game = tf.constant(team_positions_df['points_per_game'].values, dtype=tf.float32)

    match_array = [
        [team_season_to_idx.get(f"{r['league']}_{r['home_team_name']}_{r['season']}"),
         team_season_to_idx.get(f"{r['league']}_{r['away_team_name']}_{r['season']}"), r['PC1'], r['PC2']]
        for _, r in match_positions_df.iterrows()
        if f"{r['league']}_{r['home_team_name']}_{r['season']}" in team_season_to_idx and
           f"{r['league']}_{r['away_team_name']}_{r['season']}" in team_season_to_idx
    ]
    if not match_array:
        print("警告: 无有效比赛数据进行优化")
        return [], team_positions_df.reset_index(), None, None

    match_array = np.array(match_array, dtype=np.float32)
    match_home_idx = tf.constant(match_array[:, 0], dtype=tf.int32)
    match_away_idx = tf.constant(match_array[:, 1], dtype=tf.int32)
    match_PC1 = tf.constant(match_array[:, 2], dtype=tf.float32)
    match_PC2 = tf.constant(match_array[:, 3], dtype=tf.float32)

    tf.random.set_seed(random_seed)
    rank_scale = tf.Variable(1.0, dtype=tf.float32)
    w_var = tf.Variable(np.abs(w) if w is not None else np.ones(6, dtype=np.float32) / 6, dtype=tf.float32)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_lr, decay_steps, decay_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=clipnorm)

    losses, best_loss, patience_counter = [], float('inf'), 0
    best_positions, best_rank_scale, best_w = positions.numpy().copy(), 1.0, w_var.numpy().copy()

    for i in range(iterations):
        with tf.GradientTape() as tape:
            loss = compute_total_loss(positions, match_home_idx, match_away_idx, match_PC1, match_PC2, points_per_game,
                                      rank_scale, ratios, w_var, normalized_defense_score, lambda_defense,
                                      lambda_supervision, lambda_reg, elo_scores, home_away_diff)
        grads = tape.gradient(loss, [positions, rank_scale, w_var])
        if any(g is None for g in grads):
            print(f"警告: 迭代 {i + 1} 梯度为None, 跳过更新")
            continue
        optimizer.apply_gradients(zip(grads, [positions, rank_scale, w_var]))

        loss_val = float(loss.numpy())
        losses.append(loss_val)

        if loss_val < best_loss:
            best_loss, patience_counter = loss_val, 0
            best_positions, best_rank_scale, best_w = positions.numpy().copy(), float(
                rank_scale.numpy()), w_var.numpy().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停触发，训练在第 {i + 1} 轮停止")
            break

        if (i + 1) % verbose_interval == 0:
            print(f"迭代 {i + 1}/{iterations}, 损失 = {loss_val:.4f}, rank_scale = {rank_scale.numpy():.4f}")

    final_pos = scaler_pos.inverse_transform(best_positions)
    for idx, team_season in enumerate(team_seasons):
        team_positions_df.loc[team_season, 'PC1'] = final_pos[idx, 0]
        team_positions_df.loc[team_season, 'PC2'] = final_pos[idx, 1]

    return losses, team_positions_df.reset_index(), best_rank_scale, best_w


def random_search_hyperparameters(team_positions, match_positions, w, n_iter=10, random_state=42):
    """随机搜索最佳超参数"""
    np.random.seed(random_state)
    best_loss, best_params = float('inf'), None

    for i in range(n_iter):
        params = {
            'lambda_defense': uniform(0.01, 0.2).rvs(),
            'lambda_supervision': uniform(0.01, 0.2).rvs(),
            'lambda_reg': uniform(0.001, 0.02).rvs()
        }
        print(f"\n随机搜索轮次 {i + 1}/{n_iter}: {params}")

        losses, _, _, _ = adam_optimize_positions(
            team_positions.copy(), match_positions.copy(), w=w, iterations=5000, patience=50, verbose_interval=2500,
            **params
        )
        if losses and losses[-1] < best_loss:
            best_loss = losses[-1]
            best_params = params

    print(f"\n最佳超参数: {best_params}, 最佳损失: {best_loss:.4f}")
    return best_params