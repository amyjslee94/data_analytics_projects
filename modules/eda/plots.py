
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

class Exploratory():
    def __init__(self, df, figsize : tuple = (10,5), ncol = 3, x='all', y=None, hue=None ):
        self.figsize = figsize
        self.ncol = ncol
        self.df = df
        self.x = x
        self.y = y
        self.hue = hue

    def _adjust_dtypes(self, df):
        # checks nunique of integer columns and change column dtype to 'object' 
        # if nunique is less or equal to 10 for better visualization of the data
        out_df = df.copy()
        int_col_value_count = df.select_dtypes(include='int64').nunique()
        for col in int_col_value_count[int_col_value_count<=10].index: 
            out_df[col] = pd.Categorical(out_df[col], categories=sorted(list(set(out_df[col]))),ordered=True)
        return out_df
    
    def _get_target_col(self, df, col):
        if not isinstance(col, list):
            col = [col]
        if col == ['all']:
            target_col = df.columns.to_list()
        elif col == [None]:
            target_col = []
        else:
            target_col = list(set(df.columns).intersection(set(col)))
        return sorted(target_col, key=str.casefold)
        
    def _group_barplot(self, data, x, hue):
        topn = 10 if hue is None else 10 * data[hue].nunique() ## making barplot to show only the top 10 by count
        target_col = list(set([x for x in [x, hue] if x is not None])) 
        count = data[target_col].value_counts()[:topn].sort_index().reset_index(name='count')
        ax = sns.barplot(data=count, hue=hue, x=x, y='count')
        for container in ax.containers:
            ax.bar_label(container, fontsize=8)
        return ax

    def _group_stack_barplot(self, data, x, y, hue):
        topn = 10 * data[y].nunique() * data[hue].nunique()
        target_col = list(set([x for x in [x, y, hue] if x is not None]))
        count = data[target_col].value_counts()[:topn].to_frame().sort_index(ascending=False)
        count['cs'] = count.groupby([x, y], observed=False).cumsum()
        count.reset_index(inplace=True)
        
        ## for every possible groups of x,y,hue, if the data doesn't exist, it returns 0.
        unique_col1, unique_col2, unique_col3 = count[x].unique(), count[y].unique(), count[hue].unique()
        cart_prod = pd.MultiIndex.from_product(
            [unique_col1, unique_col2, unique_col3], names=[x, y, hue]
        ).to_frame(index=False, allow_duplicates=True) 
        count = cart_prod.merge(count, on=[x, y, hue], how='left')
        count[['count','cs']] = count[['count','cs']].fillna(0).astype(int)

        palette = ['pastel',"deep","muted","bright","dark"] * data[hue].nunique()
        custom_legends, true_val = [], []

        for i,g in enumerate(count.groupby(hue)):
            sorted_data=g[1].sort_values(by=[x, y], ascending=True)
            ax = sns.barplot(data = sorted_data, x = x, y='cs', hue=y, palette=palette[i], edgecolor='k')

            for g1 in sorted_data.groupby(y, observed=False):
                custom_legends.append(f"{y}={''.join(map(str, g1[1][y].unique()))} & {hue}={''.join(map(str, g1[1][hue].unique()))}")
                true_val.append(list(g1[1]['count']))
        
        for i, container in enumerate(ax.containers):
            va_position, y_const = ('bottom', 2) if i<count[y].nunique() else ('top', -2) ## top bar gets label on top while every other will get the label on the bottom
            for bar, label in zip(container, true_val[i]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + y_const, str(label),
                    ha='center', va=va_position, fontsize=8
                )
        
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, custom_legends, loc='upper left', fontsize=8)
        return ax

    def plot(self):
        fig = plt.figure(figsize=self.figsize)
        dtype_mod_df = self._adjust_dtypes(self.df)
        if self.hue is not None:
            dtype_mod_df = dtype_mod_df.sort_values(by=self.hue)
        
        target_xcol = self._get_target_col(dtype_mod_df, self.x)
        target_ycol = self._get_target_col(dtype_mod_df, self.y)

        if self.y == None:
            nrow = math.ceil(len(target_xcol)/self.ncol)
            for i, col in enumerate(target_xcol, start=1):
                ax = fig.add_subplot(nrow, self.ncol, i)
                if dtype_mod_df[col].dtypes in ['float','int']:
                    sns.histplot(data=dtype_mod_df, x=col, hue=self.hue, ax=ax)
                else:
                    self._group_barplot(data=dtype_mod_df, x=col, hue=self.hue)
                ax.set(xlabel = None)
                ax.set_title(col, fontsize=15)
        
        else:
            nrow, ncol = len(target_ycol), len(target_xcol)          
            for i, (xcol,ycol) in enumerate([(x,y) for x in target_xcol for y in target_ycol], start=1):
                ax=fig.add_subplot(nrow, ncol, i)
                if xcol!=ycol:
                    if all(dtype_mod_df[col].dtypes in ['float','int'] for col in [xcol,ycol]):
                        sns.scatterplot(data=dtype_mod_df, x=xcol, y=ycol, hue=self.hue, ax=ax)
                    elif all(dtype_mod_df[col].dtypes in ['object','category'] for col in [xcol,ycol]):
                        if self.hue in [None, ycol, xcol]:
                            self._group_barplot(data = dtype_mod_df, x = xcol, hue = ycol)
                        else:
                            self._group_stack_barplot(data=dtype_mod_df, x=xcol, y=ycol, hue=self.hue)
                    else:
                        print(dtype_mod_df[[xcol, ycol]].dtypes)
                        sns.violinplot(data=dtype_mod_df, x=xcol, y=ycol, hue=self.hue, split=True, gap=0.1, inner='quartile', ax=ax)
                        for l in ax.lines[1::3]:
                            l.set_color('red')
                else:
                    sns.scatterplot(data=dtype_mod_df, x=xcol, y=ycol, hue=self.hue, ax=ax)
                ax.set(xlabel = None, ylabel=None)
                ax.set_title(f'{xcol} vs {ycol}', fontsize=15)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    df = pd.read_csv('./Titanic/data/train.csv')
    def get_cabin_share_flg(X):
        count_df = X.value_counts()
        merged = pd.merge(X, count_df, how='left', on=['Cabin'])
        merged['transformed'] = ["private" if x==1 else "shared" if x>1 else 'no_cabin' for x in merged.iloc[:,-1]]
        return merged[['transformed']]

    df['cabin_share_flg']=get_cabin_share_flg(df[['Cabin']])
    df['cabin_share_flg']=pd.Categorical(df['cabin_share_flg'],
                                         categories=['private','shared','no_cabin'],
                                         ordered=True)

    import numpy as np
    import re
    def get_cabin_flg(X):
        X = X.copy()
        X['transformed'] = ["no_cabin" if x is np.nan else "has_cabin" for x in X.iloc[:,0]]
        return X[['transformed']]

    df['cabin_flg'] = get_cabin_flg(df[['Cabin']])

    def get_name_title(X):
        X = X.copy()
        X['transformed'] = [re.search(r", (.*?)\.", x).group(1) for x in X.iloc[:,0]]
        X['transformed'] = pd.Categorical(X['transformed'],
                                        categories=sorted(list(set(X['transformed']))),
                                        ordered=True)
        return X[['transformed']]

    df['name_title'] = get_name_title(df[['Name']])

    def get_cabin_count(X):
        X = X.copy()
        X['transformed'] = [0 if x is np.nan else len(x.split(' ')) for x in X.iloc[:,0]]
        return X[['transformed']]

    df['cabin_count'] = get_cabin_count(df[['Cabin']])
    # df['Parch'] = pd.Categorical(df['Parch'], categories=sorted(list(set(df['Parch']))),ordered=True)

    feature_columns = list(set(df.columns).difference(set(['PassengerId','Name','Ticket'])))
    # t = df['Cabin'].value_counts()[:10].sort_index()
    # sns.barplot(data = t)
    # plt.show()
    Exploratory(df = df[feature_columns], figsize = (10,10), ncol = 4, x=['Age','Sex'], y=['Parch','Embarked'], hue='Survived').plot()
    # dict = {['a','b']:['a'], ['c','d','f']:['b']}

        