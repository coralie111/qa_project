import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

train = pd.read_csv('Data/train.csv')

train.info()

g = sns.countplot('category', data=train)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
g.set_title("instance's category distribution")
plt.show()


g = sns.countplot('host', data=train, order=train.host.value_counts().iloc[:6].index)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
g.set_title("instance's category distribution")
plt.show()

# Distribution des valeurs des variables cibles

train_d = train.copy()
targets = list(train.columns[11:])

targets_question = list(train.columns[11:32])
targets_answer = list(train.columns[33:])

features = list(train.columns[:11])

train_d[targets] = train[targets].apply(lambda x: pd.cut(x,
                                                         [-0.1, 0.25, 0.5, 0.75, 1.1],
                                                         labels=['mauvais', 'moyen', 'moyen +', 'bon']))

train_d_m = train_d.melt(features)

g = sns.catplot('value', col='variable', data=train_d_m[train_d_m.variable.isin(targets_answer)],
                kind='count', col_wrap=6)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('answer type target countplot')
plt.show()

g = sns.catplot('value', col='variable', data=train_d_m[train_d_m.variable.isin(targets_question)],
                kind='count', col_wrap=6)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('question type target countplot')
plt.show()

chunk_size = 3

for i in range(0, len(targets_answer), chunk_size):
    g = sns.catplot('value', row='category', col='variable',
                    data=train_d_m[train_d_m.variable.isin(targets_answer[i:i + chunk_size])],
                    kind='count')
    plt.show()


for i in range(0, len(targets_question), chunk_size):
    g = sns.catplot('value', row='category', col='variable',
                    data=train_d_m[train_d_m.variable.isin(targets_question[i:i + chunk_size])],
                    kind='count')
    plt.show()

# fig = plt.figure()

# for i in list(range(30)):
#     fig.add_subplot(30, 1, i+1)
#     sns.catplot(x=targets[i], data=train_d, kind='count')
#     plt.title(targets[i])
# plt.show()

# # En fonction des catégories

# fig = plt.figure()

# for i in list(range(30)):
#     fig.add_subplot(30, 1, i+1)
#     sns.catplot(x=targets[i], col='category', data=train_d, kind='count')
#     plt.title(targets[i])
#     plt.show()

# # Valeurs des variables cibles (question/answer) en fonction des catégories

# targets_question = list(train.columns[11:32])
# targets_answer = list(train.columns[33:])

# fig = plt.figure(figsize = (7, 200))

# for i in list(range(len(targets_question))):
#     fig.add_subplot(30, 1, i+1)
#     sns.catplot(x='category', y=targets_question[i], kind='bar', data=train)
#     plt.show()

# fig2= plt.figure(figsize = (7, 200))

# for i in list(range(len(targets_answer))):
#     fig2.add_subplot(30, 1, i+1)
#     sns.catplot(x='category', y=targets_answer[i], kind='bar', data=train)
#     plt.show()




