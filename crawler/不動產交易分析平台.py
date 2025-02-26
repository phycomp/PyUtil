import streamlit as st pageConfig title as stTitle, sidebar, header, file_uploader flUpldr, columns as stCLMN, multiselect plotly_chart, metric
import pandas as pd read_csv, DataFrame, date_range
from plotly.express import line, box, scatter
from datetime import datetime
from numpy.random import choice as rndmChoice, normal as rndmNormal

# 設置頁面配置
pageConfig( page_title="不動產交易分析平台", page_icon="🏠", layout="wide")

# 主標題
stTitle("不動產交易分析平台 🏠")

# 側邊欄
with sidebar:
    header("數據篩選")

    uploaded_file = flUpldr("上傳CSV檔案", type="csv") # 假設我們有示例數據

    if uploaded_file is not None:
        df = read_csv(uploaded_file)
    else:
        # 創建示例數據
        dates = date_range(start='2023-01-01', end='2023-12-31', periods=100)
        df = DataFrame({'交易日期':dates,
        '區域': rndmChoice(['台北市', '新北市', '桃園市'], 100),
        '建物類型': rndmChoice(['住宅大樓', '公寓', '透天厝'], 100),
        '總價': rndmNormal(2000, 500, 100) * 10000,
        '建物面積': rndmNormal(30, 10, 100),
        '單價': rndmNormal(60, 10, 100) * 10000
        })

    # 篩選條件
    selected_area = multiselect( '選擇區域',
        options=sorted(df['區域'].unique()),
        default=sorted(df['區域'].unique()))

    selected_type = multiselect(
        '選擇建物類型',
        options=sorted(df['建物類型'].unique()),
        default=sorted(df['建物類型'].unique())
    )

    price_range = st.slider(
        '總價範圍 (萬元)',
        float(df['總價'].min())/10000,
        float(df['總價'].max())/10000,
        (float(df['總價'].min())/10000, float(df['總價'].max())/10000)
    )

# 數據篩選
filtered_df = df[
    (df['區域'].isin(selected_area)) &
    (df['建物類型'].isin(selected_type)) &
    (df['總價'] >= price_range[0]*10000) &
    (df['總價'] <= price_range[1]*10000)
]

# 主要內容區域
col1, col2 = stCLMN(2)
with col1:
    header("價格趨勢")
    fig_trend = line(
        filtered_df.groupby('交易日期')['總價'].mean().reset_index(),
        x='交易日期',
        y='總價',
        title='平均總價走勢'
    )
    plotly_chart(fig_trend, use_container_width=True)

with col2:
    header("建物類型分析")
    fig_type=box(filtered_df, x='建物類型', y='單價', title='各類型建物單價分布')
    plotly_chart(fig_type, use_container_width=True)

header("區域分析") # 區域分析
col3, col4 = stCLMN(2)

with col3:
    area_stats = filtered_df.groupby('區域').agg({'總價':'mean', '建物面積':'mean', '單價':'mean'}).round(2)

    dataframe(area_stats, use_container_width=True)

with col4:
    fig_area = scatter(filtered_df, x='建物面積', y='總價', color='區域', title='面積-總價分布', trendline="ols")
    plotly_chart(fig_area, use_container_width=True)

header("交易明細") # 詳細交易資料
dataframe(filtered_df.sort_values('交易日期', ascending=False), use_container_width=True)

header("統計摘要") # 數據統計摘要
col5, col6, col7 = stCLMN(3)

with col5:
    metric(label="平均總價", value=f"{filtered_df['總價'].mean()/10000:.2f}萬", delta=f"{filtered_df['總價'].mean()/10000 - df['總價'].mean()/10000:.2f}萬")

with col6:
    metric(label="平均單價", value=f"{filtered_df['單價'].mean()/10000:.2f}萬/坪", delta=f"{filtered_df['單價'].mean()/10000 - df['單價'].mean()/10000:.2f}萬/坪")

with col7:
    metric(label="平均面積", value=f"{filtered_df['建物面積'].mean():.2f}坪", delta=f"{filtered_df['建物面積'].mean() - df['建物面積'].mean():.2f}坪")
