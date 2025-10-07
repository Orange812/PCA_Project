
import pandas as pd
import numpy as np
import glob
import os

def load_all_matches_from_csvs(data_dir):
    """
    从 data/ 目录下的所有 -matches-*.csv 文件加载比赛数据。
    """
    csv_files = glob.glob(os.path.join(data_dir, '*-matches-*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"在 {data_dir} 目录下没有找到比赛的CSV文件。")
    
    df_list = []
    required_columns = [
        'date_GMT', 'home_team_name', 'away_team_name', 
        'home_team_corner_count', 'away_team_corner_count'
    ]
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if all(col in df.columns for col in required_columns):
                # 数据清洗：将-1替换为NaN
                df['home_team_corner_count'] = df['home_team_corner_count'].replace(-1, np.nan)
                df['away_team_corner_count'] = df['away_team_corner_count'].replace(-1, np.nan)
                df_list.append(df[required_columns])
            else:
                print(f"警告: 文件 {file} 缺少必需的列，已跳过。")
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            
    if not df_list:
        raise ValueError("没有成功加载任何包含必需列的数据。")
        
    return pd.concat(df_list, ignore_index=True)

def calculate_corner_features(df, window_size=10):
    """
    计算角球相关的滚动平均特征。
    """
    df['date_GMT'] = pd.to_datetime(df['date_GMT'])
    df = df.sort_values('date_GMT')

    df['home_avg_corners_for'] = df.groupby('home_team_name')['home_team_corner_count'].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
    )
    df['home_avg_corners_against'] = df.groupby('home_team_name')['away_team_corner_count'].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
    )
    df['away_avg_corners_for'] = df.groupby('away_team_name')['away_team_corner_count'].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
    )
    df['away_avg_corners_against'] = df.groupby('away_team_name')['home_team_corner_count'].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
    )
    
    df['total_corner_count'] = df['home_team_corner_count'] + df['away_team_corner_count']

    return df

if __name__ == '__main__':
    data_directory = 'data'
    try:
        matches_df = load_all_matches_from_csvs(data_directory)
        print(f"成功从 {len(glob.glob(os.path.join(data_directory, '*-matches-*.csv')))} 个CSV文件中加载了 {len(matches_df)} 条比赛记录。")
        
        # 在计算特征前就丢弃没有角球数据的行
        matches_df.dropna(subset=['home_team_corner_count', 'away_team_corner_count'], inplace=True)
        print(f"数据清洗后，剩余 {len(matches_df)} 条有效角球记录。")

        matches_with_features = calculate_corner_features(matches_df)
        
        # 再次丢弃因为滚动窗口而产生的NaN
        matches_with_features.dropna(subset=[
            'home_avg_corners_for', 'home_avg_corners_against',
            'away_avg_corners_for', 'away_avg_corners_against'
        ], inplace=True)

        print(f"成功计算角球特征。最终剩余 {len(matches_with_features)} 条有效记录用于建模。")
        
        feature_columns = [
            'home_team_name', 'away_team_name', 
            'home_avg_corners_for', 'home_avg_corners_against',
            'away_avg_corners_for', 'away_avg_corners_against',
            'total_corner_count'
        ]
        print(matches_with_features[feature_columns].tail())

    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
