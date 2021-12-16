from helpers.eda import *
from helpers.data_prep import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import RobustScaler


pd.set_option('display.width', 250)
pd.set_option('display.max_columns', None)
df = pd.read_csv("dataset/titanic.csv")
df.columns = [col.upper() for col in df.columns]

def titanic_data_prep(df):
    df.columns = [col.upper() for col in df.columns]
    # Cabin bool
    df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
    #now it's not boolean it's binary.
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
    #############################################

    df.head()

    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                   and df[col].nunique() == 2]

    for col in binary_cols:
        df = label_encoder(df, col)

    #############################################
    # 5. Rare Encoding
    #############################################

    rare_analyser(df, "SURVIVED", cat_cols)

    df = rare_encoder(df, 0.01, cat_cols)

    #############################################
    # 6. One-Hot Encoding
    #############################################

    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

    df = one_hot_encoder(df, ohe_cols)
    df.head()
    df.shape

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    rare_analyser(df, "SURVIVED", cat_cols)

    useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                    (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

    # df.drop(useless_cols, axis=1, inplace=True)

    #############################################
    # 7. Standart Scaler
    #############################################

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    df[num_cols].head()

    df.head()
    df.tail()

    # cat_cols, num_cols, cat_but_car = grab_col_names(df)
    return df

dff = titanic_data_prep(df)

dff.head()

#########################
# Modelleme
#########################

dff.drop(["PASSENGERID", "SIBSP_Rare", "NEW_FAMILY_SIZE_Rare", "NEW_NAME_WORD_COUNT_Rare","PARCH_Rare"], inplace = True, axis = 1)

x = dff["SURVIVED"]
y = df.drop(["SURVIVED"], axis = True)

x = dff.drop(['SURVIVED'],axis=1)
y = dff['SURVIVED']

log_model = LogisticRegression(max_iter= 10000).fit(x, y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(x)

y_pred[0:10]
y[0:10]

# Confusion Matrix
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

# Başarı Skorları
print(classification_report(y, y_pred))

# ROC-AUC
y_prob = log_model.predict_proba(x)[:, 1]
roc_auc_score(y, y_prob)

# Veri Setinin Train Seti Olarak Ayrılması

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.20, random_state=17)

# Modelin Train Setine Kurulması

log_model = LogisticRegression(max_iter=1000).fit(x_train, y_train)

# Test setinin modele sorulması:
y_pred = log_model.predict(x_test)

# AUC Score için y_prob (1. sınıfa ait olma olasılıkları)
y_prob = log_model.predict_proba(x_test)[:, 1]

# Classification Sonucu
print(classification_report(y_test, y_pred))

# ROC Curve
plot_roc_curve(log_model, x_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

roc_auc_score(y_test, y_prob)


log_model = LogisticRegression(max_iter=1000).fit(x, y)

cv_results = cross_validate(log_model,
                            x, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

# Accuracy
cv_results['test_accuracy'].mean()

# Precision
cv_results['test_precision'].mean()

# Recall
cv_results['test_recall'].mean()


# F1-score
cv_results['test_f1'].mean()


# AUC
cv_results['test_roc_auc'].mean()

