import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def transform(df):
    # SELECT NUMERIC FEATURES
    columnas_quedar = ['GrLivArea','GarageArea', 'TotalBsmtSF',
                       '1stFlrSF','TotRmsAbvGrd', 'YearBuilt',
                       'OverallQual','GarageCars', 'FullBath']

    df_transformed = df[columnas_quedar]
    
    return df_transformed


#EDA
def eda(df_train):
    # Histogram of SalePrice
    y = df_train['SalePrice']
    plt.figure(figsize=(10, 6))  
    sns.histplot(y, bins=80, kde=True, legend=False)  
    plt.title('Histogram - Sale Price (y)',fontweight='bold', size=15, y=1.05)
    plt.xticks(rotation=90)
    plt.show()

    # Heatmap of correlations
    list_num = df_train._get_numeric_data().columns
    df_train_num = df_train[list_num]
    corrmat = df_train_num.corr() 
    k = 10 
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df_train_num[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                     yticklabels=cols.values, xticklabels=cols.values)
    plt.title('Heatmap of Correlations', fontweight='bold', size=15, y=1.05)
    plt.show()

    # Histograms of top correlated features with SalePrice
    plt.figure(figsize=(18, 18))
    for i, x in enumerate(cols[1:]):
        plt.subplot(3, 3, i+1)
        sns.histplot(df_train_num[x], bins=20, kde=True, legend=False)
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.suptitle('Histograms of top correlated features with SalePrice', fontweight='bold', size=30, y=1.05)
    plt.show()

    # Scatter plots of selected features vs. SalePrice
    plt.figure(figsize=(18, 12))  
    for i, x in enumerate(['GrLivArea', 'GarageArea', 'TotalBsmtSF', 
                           '1stFlrSF', 'YearBuilt', 'YearRemodAdd'], start=1):
        plt.subplot(2, 3, i) 
        sns.scatterplot(x=df_train_num[x], y=df_train_num['SalePrice']) 
        plt.xlabel(x)
        plt.ylabel('SalePrice')
        plt.xticks(rotation=90)
        plt.tight_layout() 
    plt.suptitle('Scatter plots of selected numeric features vs. SalePrice', fontweight='bold', size=30, y=1.05)    
    plt.show()

    # Box plots of selected categorical features vs. SalePrice
    plt.figure(figsize=(18, 6))  
    for i, x in enumerate(['OverallQual', 'GarageCars', 'FullBath'], start=1):
        plt.subplot(1, 3, i) 
        sns.boxplot(x=df_train_num[x], y=df_train_num['SalePrice']) 
        #plt.title(x, fontweight='bold', size=20)
        plt.xlabel(x)
        plt.ylabel('SalePrice')
        plt.xticks(rotation=90)
        plt.tight_layout() 
    plt.suptitle('Box plots of selected categorical features vs. SalePrice', fontweight='bold', size=30, y=1.05)
    plt.show()