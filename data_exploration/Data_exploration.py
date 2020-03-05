import numpy as np
import pandas as pd

fn_list = [lambda x: x.name,
           lambda x: np.where(get_column_index(x) > 10, 'target', 'feature'),
           lambda x: '',
           lambda x: '',
           lambda x: x.dtype,
           lambda x: x.isna().sum() / len(x),
           lambda x: '',
           lambda x: x.describe().to_string(),
           lambda x: etendue_valeurs(x),
           lambda x: '',
           lambda x: x.value_counts().iloc[:10].to_string()
           ]


def get_column_index(input_series):
    return data.columns.get_loc(input_series.name)


to_exclude_column = ['qa_id', 'question_title', 'question_body',
                     'question_user_name', 'answer_user_name', 'answer']


def etendue_valeurs(input_series):
    if input_series.name in to_exclude_column:
        return 'NA'
    if input_series.dtype == 'object':
        return '\n'.join(input_series.unique())
    else:
        return 'min: {:f}; max: {:f}'.format(input_series.min(), input_series.max())


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

data = pd.concat([train, test])

output_l = list()

for fn in fn_list:
    result = data.apply(fn, result_type='expand')
    if isinstance(result, pd.Series):
        output_l.append(result.to_frame().T)
    else:
        output_l.append(result)

pd.concat(output_l).T.to_csv('../output/output.csv', sep='|')
