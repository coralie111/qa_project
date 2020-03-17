import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


train = pd.read_csv('data/train.csv')

to_delete_var = ['qa_id', 'url',
                 'question_user_name', 'question_user_page',
                 'answer_user_name', 'answer_user_page']

X_train, y_train = train.iloc[:, :11].drop(to_delete_var, 1), train.iloc[:, 11:]

y_train_transformed = y_train.apply(lambda x: pd.cut(x,
                                                     [-0.1, .25, .5, .75, 1.1],
                                                     labels=['low', 'medium-', 'medium+', 'high']))

vectorizer = CountVectorizer(stop_words='english',
                             lowercase=True,
                             ngram_range=(1, 2))
ohe = OneHotEncoder(drop='first', sparse=True)

text_var = ['question_title', 'question_body', 'answer']

X_train_text = sp.hstack(X_train[text_var].apply(lambda col: vectorizer.fit_transform(col)))
X_train_category = ohe.fit_transform(X_train[['category', 'host']])
X_train_transformed = sp.hstack([X_train_text, X_train_category])

X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train_transformed, y_train_transformed, test_size=0.2)

dtc = DecisionTreeClassifier(max_depth=5).fit(X_train_train, y_train_train)

dtc.score(X_train_test, y_train_test)

# tree interpretation

# feats = {}
# for feature, importance in zip(XXX.columns, dtc.feature_importances_):
#     feats[feature] = importance
# importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
# importances.sort_values(by='Importance', ascending = False ).head(8)

# affichage de l'arbre de décision

# fig, ax = plt.subplots(figsize=(20, 20))
# sklearn.tree.plot_tree(dtc, ax=ax, filled=True)
