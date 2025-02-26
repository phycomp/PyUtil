import streamlit as st, set_page_config pageConfig, title as stTitle, sidebar, header, radio as stRadio, file_uploader as flUpldr, button stButton, spinner as stSpinner, success as stSuccess, columns as stCLMN, plotly_chart, metric, header, dataframe
from stUtil import rndrCode
from pandas import read_csv, DataFrame
from plotly.express import box, scatter
from glob import glob
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from house_591_crawler import get_591_data  # 導入爬蟲模組

pageConfig( page_title="591不動產交易分析平台", page_icon="🏠", layout="wide") # 設置頁面配置

# 主標題
stTitle("591不動產交易分析平台 🏠")

# 側邊欄
with sidebar:
    header("數據來源選擇")

    data_source = stRadio("選擇數據來源", ("上傳CSV檔案", "爬取591最新資料"))

    if data_source == "上傳CSV檔案":
        uploaded_file = flUpldr("上傳CSV檔案", type="csv")
        if uploaded_file is not None:
            df = read_csv(uploaded_file)
    else:
        if stButton("開始爬取591資料"):
            with stSpinner('正在爬取591房屋資料...'):
                df = get_591_data()
                stSuccess('資料爬取完成！')
        else:
            # 讀取最近的591資料檔案（如果存在）
            files = glob('591_house_data_*.csv')
            if files:
                latest_file = max(files)
                df = pd.read_csv(latest_file)
                rndrCode(f'顯示最近爬取的資料: {latest_file}')
            else:
                st.warning('尚未有爬取資料，請點擊按鈕開始爬取')
                df = pd.DataFrame()

    if not df.empty:
        # 篩選條件
        selected_city = st.multiselect(
            '選擇縣市',
            options=sorted(df['縣市'].unique()),
            default=sorted(df['縣市'].unique())
        )

        selected_type = st.multiselect(
            '選擇建物類型',
            options=sorted(df['建物類型'].unique()),
            default=sorted(df['建物類型'].unique())
        )

        price_range = st.slider(
            '總價範圍 (萬元)',
            float(df['總價'].min()),
            float(df['總價'].max()),
            (float(df['總價'].min()), float(df['總價'].max()))
        )

# 如果有資料才顯示分析內容
if not df.empty:
    # 數據篩選
    filtered_df = df[
        (df['縣市'].isin(selected_city)) &
        (df['建物類型'].isin(selected_type)) &
        (df['總價'] >= price_range[0]) &
        (df['總價'] <= price_range[1])
    ]

    # 主要內容區域
    col1, col2 = stCLMN(2)

    with col1:
        st.header("各縣市房價分布")
        fig_city = box(
            filtered_df,
            x='縣市',
            y='總價',
            title='各縣市房價分布'
        )
        st.plotly_chart(fig_city, use_container_width=True)

    with col2:
        st.header("建物類型分析")
        fig_type = px.box(
            filtered_df,
            x='建物類型',
            y='單價',
            title='各類型建物單價分布'
        )
        st.plotly_chart(fig_type, use_container_width=True)

    # 區域分析
    st.header("區域分析")
    col3, col4 = st.columns(2)

    with col3:
        area_stats = filtered_df.groupby('縣市').agg({
            '總價': 'mean',
            '建物面積': 'mean',
            '單價': 'mean'
        }).round(2)

        st.dataframe(area_stats, use_container_width=True)

    with col4:
        fig_area = scatter(filtered_df, x='建物面積', y='總價', color='縣市', title='面積-總價分布', trendline="ols")
        plotly_chart(fig_area, use_container_width=True)

    # 詳細交易資料
    header("房屋列表")
    dataframe(filtered_df.sort_values('總價', ascending=False), use_container_width=True)

    # 數據統計摘要
    header("統計摘要")
    col5, col6, col7 = stCLMN(3)
    with col5:
        metric(label="平均總價", value=f"{filtered_df['總價'].mean():.2f}萬", delta=f"{filtered_df['總價'].mean() - df['總價'].mean():.2f}萬")

    with col6:
        metric(label="平均單價", value=f"{filtered_df['單價'].mean():.2f}萬/坪", delta=f"{filtered_df['單價'].mean() - df['單價'].mean():.2f}萬/坪")

    with col7:
        metric(label="平均面積", value=f"{filtered_df['建物面積'].mean():.2f}坪", delta=f"{filtered_df['建物面積'].mean() - df['建物面積'].mean():.2f}坪")
