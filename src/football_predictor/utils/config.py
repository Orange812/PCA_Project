# config.py
# 描述: 存储项目的所有配置常量

import os

# ============== 文件路径与目录 ==============
# 请将此路径修改为您本地存放数据文件的实际路径
BASE_PATH = '/Users/peixuanma/Downloads/data1'

# 输出目录
OUTPUT_DIR = "/Users/peixuanma/Downloads/Output_Graphs"
# 最终生成的CSV文件名
FINAL_CSV_PATH = "trained_team_positions.csv"


# ============== 联赛与赛季 ==============
LEAGUES = [
    ("england", "premier-league"), ("germany", "bundesliga"), ("spain", "la-liga"),
    ("france", "ligue-1"), ("france", "ligue-2"), ("italy", "serie-a"), ("netherlands", "eredivisie"),
    ("portugal", "ligapro"), ("denmark", "superliga"), ("england", "championship"), ("spain", "segunda-division"),
    ("switzerland", "super-league"), ("portugal", "liga-nos"), ("italy", "serie-b"), ("germany", "2-bundesliga"),
    ("scotland", "premiership"), ("belgium", "pro-league"), ("austria", "bundesliga"),
]

SEASONS = [
    "2013-to-2014", "2014-to-2015", "2015-to-2016", "2016-to-2017", "2017-to-2018",
    "2018-to-2019", "2019-to-2020", "2020-to-2021", "2021-to-2022", "2022-to-2023", "2023-to-2024"
]