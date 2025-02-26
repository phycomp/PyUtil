from pandas import qcut, DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from plotly.express import scatter, parallel_coordinates, sunburst
from plotly.graph_objects import Figure, Heatmap
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
class HouseAnalyzer:
    def __init__(self, df):
        self.df = df
        self.model = None
    def preprocess_data(self):
        """資料預處理"""
        # 移除異常值
        self.df = self.df[
            (self.df['總價'] > self.df['總價'].quantile(0.01)) &
            (self.df['總價'] < self.df['總價'].quantile(0.99)) &
            (self.df['建物面積'] > self.df['建物面積'].quantile(0.01)) &
            (self.df['建物面積'] < self.df['建物面積'].quantile(0.99))
        ]
        # 計算其他特徵
        self.df['單價區間'] = qcut(self.df['單價'], q=5, labels=['低價', '中低價', '中價', '中高價', '高價'])
        self.df['面積區間'] = qcut(self.df['建物面積'], q=5, labels=['小坪數', '中小坪數', '中坪數', '中大坪數', '大坪數'])
        return self.df
    def create_price_heatmap(self):
        """創建價格熱力圖"""
        pivot_table = self.df.pivot_table(
            values='單價',
            index='縣市',
            columns='建物類型',
            aggfunc='mean'
        )
        fig = Figure(data=Heatmap(z=pivot_table.values, x=pivot_table.columns, y=pivot_table.index, colorscale='Viridis'))
        fig.update_layout(title='各縣市建物類型單價熱力圖', xaxis_title='建物類型', yaxis_title='縣市')
        return fig
    def create_bubble_chart(self):
        """創建泡泡圖"""
        city_stats = self.df.groupby('縣市').agg({
            '總價': 'mean',
            '建物面積': 'mean',
            '單價': 'mean',
            '標題': 'count'
        }).reset_index()
        fig = scatter( city_stats, x='建物面積', y='單價', size='標題', color='縣市', hover_data=['總價'], title='各縣市房價分布泡泡圖')
        return fig
    def create_parallel_coordinates(self):
        """創建平行座標圖"""
        fig = parallel_coordinates(
            self.df,
            dimensions=['總價', '建物面積', '單價'],
            color='縣市',
            title='房屋特徵平行座標圖'
        )
        return fig
    def create_sunburst(self):
        """創建旭日圖"""
        fig = sunburst(self.df, path=['縣市', '建物類型', '單價區間'], values='總價', title='房屋市場結構旭日圖')
        return fig
    def train_price_predictor(self):
        """訓練價格預測模型"""
        # 準備特徵
        features = ['建物面積', '縣市', '建物類型']
        X = self.df[features]
        y = self.df['總價']
        # 分割訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # 建立預處理管道
        numeric_features = ['建物面積']
        categorical_features = ['縣市', '建物類型']
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
            ]
        )
        # 建立模型管道
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        # 訓練模型
        self.model.fit(X_train, y_train)
        # 計算準確度
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        return train_score, test_score
    def predict_price(self, features_dict):
        """預測房屋價格"""
        if self.model is None:
            self.train_price_predictor()
        # 創建特徵DataFrame
        features_df = DataFrame([features_dict])
        # 預測價格
        predicted_price = self.model.predict(features_df)[0]
        return predicted_price
    def get_market_insights(self):
        """獲取市場分析洞察"""
        insights = {
            '平均價格': { '整體': self.df['總價'].mean(), '各縣市': self.df.groupby('縣市')['總價'].mean().to_dict() },
            '價格區間分布': self.df['單價區間'].value_counts().to_dict(),
            '建物類型分布': self.df['建物類型'].value_
