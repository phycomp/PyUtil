import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from scipy.stats import chi2_contingency

# EDA Framework
class EDAFramework:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = data.select_dtypes(include=[object, 'category']).columns.tolist()

    # Step 1: Basic Info
    def basic_info(self):
        print("Basic Information")
        print(self.data.info())
        print("\nSummary Statistics")
        print(self.data.describe())

    # Step 2: Missing Values
    def missing_values(self):
        print("\nMissing Values")
        missing = self.data.isnull().sum()
        print(missing[missing > 0])
        sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()

    # Step 3: Data Distribution
    def data_distribution(self):
        print("\nData Distribution")
        for feature in self.numeric_features:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.data[feature], kde=True)
            plt.title(f'Distribution of {feature}')
            plt.show()

    # Step 4: Target Variable Distribution
    def target_distribution(self):
        print("\nTarget Variable Distribution")
        target_counts = self.data[self.target_column].value_counts()
        sns.barplot(x=target_counts.index, y=target_counts.values)
        plt.title(f'Distribution of Target Variable: {self.target_column}')
        plt.show()

    # Step 5: Numerical Features vs Target
    def numerical_vs_target(self):
        print("\nNumerical Features vs Target")
        for feature in self.numeric_features:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.data[self.target_column], y=self.data[feature])
            plt.title(f'{feature} vs {self.target_column}')
            plt.show()

    # Step 6: Categorical Features vs Target
    def categorical_vs_target(self):
        print("\nCategorical Features vs Target")
        for feature in self.categorical_features:
            if feature != self.target_column:
                contingency = pd.crosstab(self.data[feature], self.data[self.target_column])
                chi2, p, dof, ex = chi2_contingency(contingency)
                print(f"Chi-square test for {feature}: p-value = {p}")
                sns.countplot(x=feature, hue=self.target_column, data=self.data)
                plt.title(f'{feature} vs {self.target_column}')
                plt.show()

    # Step 7: Correlation Analysis
    def correlation_analysis(self):
        print("\nCorrelation Analysis (Numerical Features)")
        corr_matrix = self.data[self.numeric_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    # Step 8: Mutual Information (Feature Selection)
    def feature_selection(self):
        print("\nMutual Information Analysis")
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        mi_scores = mutual_info_classif(X[self.numeric_features], y)
        mi_df = pd.DataFrame({'Feature': self.numeric_features, 'Mutual Information': mi_scores})
        mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)
        print(mi_df)

    # Step 9: Recursive Feature Elimination (RFE)
    def recursive_feature_elimination(self):
        print("\nRecursive Feature Elimination")
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        model = RandomForestClassifier()
        rfe = RFE(model, n_features_to_select=5)
        X_rfe = rfe.fit_transform(X[self.numeric_features], y)

        rfe_features = pd.DataFrame({'Feature': X[self.numeric_features].columns, 'Selected': rfe.support_})
        print(rfe_features[rfe_features['Selected'] == True])

    # Run All EDA Steps
    def run_all(self):
        self.basic_info()
        self.missing_values()
        self.data_distribution()
        self.target_distribution()
        self.numerical_vs_target()
        self.categorical_vs_target()
        self.correlation_analysis()
        self.feature_selection()
        self.recursive_feature_elimination()

# 使用範例：
if __name__ == "__main__":
    # 讀取資料集 (CSV)
    data = pd.read_csv("your_dataset.csv")

    # 指定目標變數 (target_column)
    eda = EDAFramework(data, target_column='your_target_column')

    # 執行所有 EDA 步驟
    eda.run_all()
