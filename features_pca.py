import os
import sklearn
import pandas as pd
import joblib

from paths import (plots_dir, joblib_dir)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# =========================
overwrite_joblib = False
# =========================

train = pd.read_csv('data/train.csv')

X = train.iloc[:, :11].drop(to_delete_var, 1)
X_question = train.question_body
X_answer = train.answer


answer_tfidftransformer = TfidfVectorizer()

answer_tfidftransformed = answer_tfidftransformer.fit_transform(X_answer)
question_tfidftransformed = answer_tfidftransformer.transform(X_question)
title_tfidftransformed = answer_tfidftransformer.transform(X_title)

answer_pca = TruncatedSVD(n_components=10000)
answer_pca.fit(answer_tfidftransformed)

plt.figure()
plt.xlim(0,10000)
plt.axhline(y = 0.9, color ='r', linestyle = '--')
plt.plot(answer_pca.explained_variance_ratio_.cumsum());
plt.title('answer features pca')
plt.savefig(plots_dir+'answer_features_pca')

answer_joblib_fn = 'answer_pca.joblib'
if ((not os.path.isfile(answer_joblib_fn)) and overwrite_joblib):
    joblib.dump(answer_pca, joblib_dir+answer_joblib_fn)

question_pca = TruncatedSVD(n_components=10000)
question_pca.fit(question_tfidftransformed)

plt.figure()
plt.xlim(0,10000)
plt.axhline(y = 0.9, color ='r', linestyle = '--')
plt.plot(question_pca.explained_variance_ratio_.cumsum());
plt.title('question features pca')
plt.savefig(plots_dir+'question_features_pca')

question_joblib_fn = 'question_pca.joblib'
if ((not os.path.isfile(question_joblib_fn)) and overwrite_joblib):
    joblib.dump(question_pca, joblib_dir+question_joblib_fn)
    
# best_features_question = [vocab[i] for i in pca2.components_[0].argsort()[::-1]]    