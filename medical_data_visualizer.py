import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['bmi'] = (df['weight'] / (df['height'] ** 2)) * 100
df['overweight'] = (df['bmi'] > 25).astype(int)

# 3
columns = ['ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'overweight']
for column in columns:
    df[column] = df[column].apply(lambda x: 1 if x > 1 else x)

# 4
def draw_cat_plot():

    # 5
    df_cat = pd.melt(df[columns], var_name='Category', value_name='Value')

    # 6
    df_cat = pd.melt(df[columns], var_name='Category', value_name='Value')
    df_cat['Cardio'] = df['cardio'].repeat(len(df))
    df_grouped = df_cat.groupby(['Cardio', 'Category', 'Value']).size().reset_index(name='Count')

    # 7
    sns.catplot(x='Category', hue='Value', kind='count', data=df_cat)

    # 8
    fig = sns.catplot(x='Category', hue='Value', kind='count', data=df_cat)

    # 9
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) &
            (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) &
            (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', mask=mask, ax=ax, cbar_kws={'shrink': 0.8})

    # 16
    fig.savefig('heatmap.png')
    return fig
