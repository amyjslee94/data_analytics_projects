import matplotlib.pyplot as plt
import seaborn as sns
import math

class Univariate():
    def __init__(self, figsize : tuple = (10,5), ncol = 3):
        self.figsize = figsize
        self.ncol = ncol

    def plot(self, df, hue=None):
        fig = plt.figure(figsize=self.figsize)
        nrow = math.ceil(len(df.columns)/self.ncol)
        i = 1
        for col in df.columns:
            sub = fig.add_subplot(nrow, self.ncol, i)
            if df[col].dtypes in ['float', 'int']:
                ax = sns.histplot(data=df, x=col, hue=hue)
            elif df[col].dtypes in ['object', 'category']:
                ax = sns.countplot(data=df, x=col, hue=hue, order=df[col].value_counts()[:10].index)
                for container in ax.containers:
                    ax.bar_label(container, fontsize=10)
            ax.set(xlabel=None)
            ax.set_title(col, fontsize=15)
            i+=1
        plt.tight_layout()
        plt.show()

# class Bivariate():
#     def __init__(self, figsize : tuple = (20,10), ncol = 3):
#         self.figsize = figsize
#         self.ncol = ncol
    
#     def plot(self, X, y):
#         fig = plt.figure(figsize=self.figsize)
#         nrow = 
#         nrow = math.ceil(len(df.columns)/self.ncol)
        