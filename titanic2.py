

import numpy as np
import pandas as pd

from helpers.eda import *
from helpers.data_prep import *
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold


pd.set_option('display.width', 250)
pd.set_option('display.max_columns', None)
df = pd.read_csv("dataset/titanic.csv")
df.columns = [col.upper() for col in df.columns]


def titanic_data_prep(df):
    # Cabin bool
    df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
    # Name count
    df["NEW_NAME_COUNT"] = df["NAME"].str.len()
    # name word count
    df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
    # name dr
    df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    # name title
    df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    # family size
    df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
    # age_pclass
    df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
    # is alone
    df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
    # age level
    df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    # sex x age
    df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
    df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    #############################################
    # 2. Outliers (Aykırı Değerler)
    #############################################
    for col in num_cols:
        print(col, check_outlier(df, col))

    for col in num_cols:
        replace_with_thresholds(df, col)

    for col in num_cols:
        print(col, check_outlier(df, col))

    #############################################
    # 3. Missing Values (Eksik Değerler)
    #############################################
    missing_values_table(df)
    df.drop("CABIN", inplace=True, axis=1)

    remove_cols = ["TICKET", "NAME"]
    df.drop(remove_cols, inplace=True, axis=1)
    df.head()

    missing_values_table(df)

    df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

    df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

    df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
    df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

    #############################################
    # 4. Label Encoding
    ##############################################
    df.head()

    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                   and df[col].nunique() == 2]

    for col in binary_cols:
        df = label_encoder(df, col)

    #############################################
    # 5. Rare Encoding
    ##############################################
    rare_analyser(df, "SURVIVED", cat_cols)

    df = rare_encoder(df, 0.01, cat_cols)

    #############################################
    # 6. One-Hot Encoding
    ##############################################
    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

    df = one_hot_encoder(df, ohe_cols)
    df.head()
    df.shape

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    rare_analyser(df, "SURVIVED", cat_cols)

    useless_cols = [col for col in df.columns if
                    df[col].nunique() == 2 and (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

    # df.drop(useless_cols, axis=1, inplace=True)
    #############################################
    # 7. Standart Scaler
    ##############################################
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    df[num_cols].head()

    df.head()
    df.tail()

    # cat_cols, num_cols, cat_but_car = grab_col_names(df) return df
    return df

dff = titanic_data_prep(df)