import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
import scipy.sparse as sp


df = pd.read_csv('data/train.csv')

to_delete_var = ['qa_id', 'url',
                 'question_user_name', 'question_user_page',
                 'answer_user_name', 'answer_user_page']

data, targets = df.iloc[:, :11].drop(to_delete_var, 1), df.iloc[:, 11:]

targets = targets.apply(lambda x: pd.cut(x,
                                         [-0.1, .25, .5, .75, 1.1],
                                         labels=['low', 'medium-', 'medium+', 'high']))
targets_binarized = MultiLabelBinarizer().fit_transform(targets)

vectorizer = CountVectorizer(stop_words='english',
                             lowercase=True,
                             ngram_range=(1,2))

text_var = ['question_title', 'question_body', 'answer']

text_features = sp.hstack(data[text_var].apply(lambda col: vectorizer.fit_transform(col)))
category_features = OneHotEncoder(drop='first', sparse=True).fit_transform(df[['category', 'host']])

features = sp.hstack([text_features, category_features])

clf = MultiOutputClassifier(DecisionTreeClassifier(max_depth=5)).fit(features, targets_binarized)

