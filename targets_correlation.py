import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from paths import plots_dir

train = pd.read_csv('data/train.csv')

y = train.iloc[:, 11:]

y_question = y.loc[:, y.columns.str.startswith('question')]
y_answer = y.loc[:, y.columns.str.startswith('answer')]

plt.figure(figsize=(15,15))
y_question_corr = sns.heatmap(y_question.corr(),  annot=True, cmap="RdBu_r", center =0)
plt.title('question targets correlation')
plt.savefig(plots_dir+'y_question_corr')

plt.figure(figsize=(13,13))
y_question_corr = sns.heatmap(y_answer.corr(),  annot=True, cmap="RdBu_r", center =0)
plt.title('answer targets correlation')
plt.savefig(plots_dir+'y_answer_corr')
