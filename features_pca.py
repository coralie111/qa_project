import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from paths import (plots_dir, joblib_dir)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


train = pd.read_csv('data/train.csv')

X_question = train.question_body
X_answer = train.answer


# using sublinear_tf=True to bring the feature values
# closer to a Gaussian distribution, compensating for LSA’s
# erroneous assumptions about textual data
answer_tfidftransformer = TfidfVectorizer(sublinear_tf=True)

answer_tfidftransformed = answer_tfidftransformer.fit_transform(X_answer)
question_tfidftransformed = answer_tfidftransformer.transform(X_question)

n_components = 4000
answer_pca = TruncatedSVD(n_components=n_components)
answer_pca.fit(answer_tfidftransformed)

plt.figure()
plt.xlim(0, n_components)
plt.axhline(y=0.9, color='r', linestyle='--')
plt.plot(answer_pca.explained_variance_ratio_.cumsum())
answer_plot_title = 'answer features pca'
plt.title(answer_plot_title)
answer_save_fn = '_'.join(answer_plot_title.split()) + '_' + str(n_components)
plt.savefig(plots_dir+answer_save_fn)

print('number of components for answer features: ' +
      str(np.argmax(answer_pca.explained_variance_ratio_.cumsum() > 0.9)))

answer_joblib_fn = answer_save_fn+'.joblib'
answer_joblib_path = joblib_dir+answer_joblib_fn
joblib.dump(answer_pca, answer_joblib_path)

n_components = 3000
question_pca = TruncatedSVD(n_components=n_components)
question_pca.fit(question_tfidftransformed)

plt.figure()
plt.xlim(0, n_components)
plt.axhline(y=0.9, color='r', linestyle='--')
plt.plot(question_pca.explained_variance_ratio_.cumsum())
question_plot_title = 'question features pca' 
plt.title(question_plot_title)
question_save_fn = '_'.join(question_plot_title.split()) +\
                   '_' + str(n_components)
plt.savefig(plots_dir+question_save_fn)

print('number of components for question features: ' +
      str(np.argmax(question_pca.explained_variance_ratio_.cumsum() > 0.9)))


question_joblib_fn = question_save_fn + '.joblib'
question_joblib_path = joblib_dir + question_joblib_fn
joblib.dump(question_pca, question_joblib_path)

# best_features_question = [vocab[i] for i in pca2.components_[0].argsort()[::-1]]    