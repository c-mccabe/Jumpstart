import numpy as np
from sklearn.preprocessing import OneHotEncoder

model_columns = ['In work',
                 'Years since graduation (and being on programme)',
                 'Applicant type',
                 'Academics',
                 'Extra-Curricular',
                 'Work Experience',
                 'Motivations (general)',
                 'Problem solving (gym/spotify)',
                 'Jumpstart interview score']


def preprocess_df(df):
    df.loc[df['Years since graduation (and being on programme)'] == 'Yes',
           'Years since graduation (and being on programme)'] = 1

    df.loc[df['In work'] == 'No', 'In work'] = 0
    df.loc[df['In work'] == 'Yes', 'In work'] = 1

    return df


def one_hot_encode_applicant_type(X):
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    one_hot = enc.fit_transform(X[:, 2].reshape(-1, 1))
    X = np.hstack([X[:, :2], X[:, 3:], one_hot])
    return X


def get_X_y(df):
    df = preprocess_df(df)
    X = np.array(df[model_columns].fillna(0))
    y = np.array(df['Start-up interview rating'])
    X = one_hot_encode_applicant_type(X)
    return X, y
