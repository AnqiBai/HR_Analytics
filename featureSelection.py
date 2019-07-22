from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer
from mstrio import microstrategy 
import pandas as pd
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")


def greet():
    return 'Greeting! from feature selection'

def update_FS(count_of_feature, path_to_raw_data):
    # prepare
    application_train = pd.read_csv(path_to_raw_data)
    application_train.head()
    application_train.columns

    target = 'left'
    # Stratified Sampling
    application_sample1 = application_train.loc[application_train[target] == 1].sample(
        frac=0.5, replace=False)
    print('label 1 sample size:', str(application_sample1.shape[0]))
    application_sample0 = application_train.loc[application_train[target] == 0].sample(
        frac=0.5, replace=False)
    print('label 0 sample size:', str(application_sample0.shape[0]))
    application = pd.concat([application_sample1, application_sample0], axis=0)


    # impute missing values
    categorical_list = []
    numerical_list = []

    for i in application.columns.tolist():
        if application[i].dtype == 'object':
            categorical_list.append(i)
        else:
            numerical_list.append(i)
    print('Number of categorical features:', str(len(categorical_list)))
    print('Number of numerical features:', str(len(numerical_list)))

    application[numerical_list] = Imputer(strategy='median').fit_transform(application[numerical_list])

    # deal with categorical features: OneHotEncoding
    del application_train
    gc.collect()
    application = pd.get_dummies(application, drop_first=True)
    print(application.shape)
    
    for j in application.columns:
        print(j)

    # feature matrix and target 
    #X = application.drop(['left', 'ID'], axis=1)
    X = application.drop(['left'], axis=1)
    y = application[target]
    feature_name = X.columns.tolist()

    '''
    Feature Selection
    '''
    # 1. filter
    # 1.1 Pearson Correlation 


    def cor_selector(X, y):
        cor_list = []
        # calculate the correlation with y for each feature
        for i in X.columns.tolist():
            cor = np.corrcoef(X[i], y)[0, 1]
            cor_list.append(cor)
        # replace NaN with 0
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        # feature name
        cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-count_of_feature:]].columns.tolist()
        # feature selection? 0 for not select, 1 for select
        cor_support = [True if i in cor_feature else False for i in feature_name]
        return cor_support, cor_feature

    cor_support, cor_feature = cor_selector(X, y)
    print(str(len(cor_feature)), 'selected features - Pearson')

    # 1.2 Chi-2
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=count_of_feature)
    chi_selector.fit(X_norm, y)

    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()
    print(str(len(chi_feature)), 'selected features - Chi-2')

    # 2. Wrapper
    rfe_selector = RFE(estimator=LogisticRegression(),
                       n_features_to_select=count_of_feature, step=9, verbose=5)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:, rfe_support].columns.tolist()
    print(str(len(rfe_feature)), 'selected features - RFE')

    # 3. Embeded
    # 3.1 Logistics Regression L1

    embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), '1.25*median')
    embeded_lr_selector.fit(X_norm, y)
    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:, embeded_lr_support].columns.tolist()
    print(str(len(embeded_lr_feature)), 'selected features - Logistics Regression L1')

    # 3.2 Random Forest

    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=count_of_feature), threshold='1.25*median')
    embeded_rf_selector.fit(X, y)
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:, embeded_rf_support].columns.tolist()
    print(str(len(embeded_rf_feature)), 'selected features - Random Forest')

    # 3.3 LightGBM

    lgbc = LGBMClassifier(n_estimators=count_of_feature, learning_rate=0.05, num_leaves=count_of_feature*2, colsample_bytree=0.2,
                          reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

    embeded_lgb_selector = SelectFromModel(lgbc, threshold='1.25*median')
    embeded_lgb_selector.fit(X, y)
    embeded_lgb_support = embeded_lgb_selector.get_support()

    embeded_lgb_feature = X.loc[:, embeded_lgb_support].columns.tolist()
    print(str(len(embeded_lgb_feature)), 'selected features - LightGBM')

    # Summary
    pd.set_option('display.max_rows', None)
    # put all selection together
    feature_selection_df = pd.DataFrame({'Feature': feature_name, 'Pearson': cor_support, 'Chi-2': chi_support, 'RFE': rfe_support, 'Logistics': embeded_lr_support,
                                         'Random Forest': embeded_rf_support, 'LightGBM': embeded_lgb_support})
    # count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(
        ['Total', 'Feature'], ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    feature_selection_df.head(100)

    '''
    Send the resule to microstrategy
    '''

    # home_dir = '/Users/anqib/Desktop/WorkZone/HR_analytics'

    username = 'administrator'
    password = 'chuaijiji)98'
    base_url = 'http://54.173.135.146:8080/MicroStrategyLibrary/api'
    project_name = 'Model BI'

    conn = microstrategy.Connection(base_url, username, password, project_name)
    conn.connect()

    dataset_id_to_update = 'A71524DE409F4066DDAB9EBAE2584728'
    # conn.create_dataset(data_frame=feature_selection_df, dataset_name='featureSelection', table_name='features')
    conn.update_dataset(data_frame = feature_selection_df[:count_of_feature], dataset_id = dataset_id_to_update, 
                        table_name = 'features', update_policy = 'replace')

    #log out
    conn.close()

    return 'Updated'

