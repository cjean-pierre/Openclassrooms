import pandas as pd
import numpy as np


def one_hot_encoder(df, nan_as_category=False):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def application_train_test():
    # Read data and merge
    df = pd.read_csv('./Data/application_train.csv')
    test_df = pd.read_csv('./Data/application_test.csv')
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = pd.concat([df, test_df]).reset_index(drop=True)

    # simplifying number of categories
    for i in ['Incomplete higher', 'Lower secondary', 'Academic degree']:
        df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].str.replace(i, 'Other')

    for i in ['Spouse, partner', 'Children', 'Other_B', 'Other_A', 'Group of people']:
        df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].str.replace(i, 'Other')

    for i in ['Separated', 'Widow', 'Unknown']:
        df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].str.replace(i, 'Other')
    df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].str.replace('Civil marriage', 'Married')

    for i in ['Municipal apartment', 'Rented apartment', 'Office apartment', 'Co-op apartment']:
        df['NAME_HOUSING_TYPE'] = df['NAME_HOUSING_TYPE'].str.replace(i, 'Other')

    for i in ['IT staff', 'HR staff', 'Realty agents', 'Secretaries', 'Waiters/barmen staff']:
        df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].str.replace(i, 'Other')

    for org in ["Business Entity", "Trade", 'Industry', 'Transport']:
        mask = df['ORGANIZATION_TYPE'].str.contains(org)
        df.loc[mask, 'ORGANIZATION_TYPE'] = df.loc[mask, 'ORGANIZATION_TYPE'].map(lambda x: org)

    for i in ['XNA', 'Restaurant', 'Postal', 'Telecom', 'Realtor', 'Legal Services', 'Advertising',
              'Emergency', 'Mobile', 'Religion', 'Cleaning', 'Insurance', 'Culture']:
        df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].str.replace(i, 'Other')

    # drop contact info columns
    df.drop(columns=['FLAG_MOBIL', 'FLAG_CONT_MOBILE',
                     'FLAG_PHONE', 'FLAG_EMAIL'], inplace=True)

    # df.drop(columns=['ORGANIZATION_TYPE'], inplace=True)

    # drop FLAGs on address
    flag_add_cols = ['REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
                     'LIVE_REGION_NOT_WORK_REGION']

    df.drop(columns=flag_add_cols, inplace=True)

    # drop living info columns
    df.drop(columns=['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
                     'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ENTRANCES_AVG',
                     'FLOORSMIN_AVG', 'LANDAREA_AVG',
                     'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
                     'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE',
                     'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE',
                     'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE',
                     'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
                     'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
                     'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
                     'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI',
                     'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
                     'LIVINGAREA_MEDI',
                     'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE',
                     'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'], inplace=True)

    # replace FLAG_DOCUMENT_X columns total documents
    flag_cols = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
                 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
                 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
                 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
                 'FLAG_DOCUMENT_21']
    df['TOTAL_DOCUMENTS'] = df.loc[:, flag_cols].sum(axis=1)
    df.drop(columns=flag_cols, inplace=True)

    # replace AMT_REG_CREDIT_BUREAU columns with total
    amt_reg_cols = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                    'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
    df['TOTAL_AMT_REQ_CB'] = df.loc[:, amt_reg_cols].sum(axis=1)
    df.drop(columns=amt_reg_cols, inplace=True)

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category=False)

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    df.columns = pd.Index(['SK_ID_CURR', 'TARGET']+('APPLI_' + df.columns.str.upper()).to_list()[2:])
    del test_df

    return df
