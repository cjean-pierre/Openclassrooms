import pandas as pd
import numpy as np
from pathlib import Path

def previous_applications():
    """ perform preprocessing of previous applications """
    path = Path(__file__).parent
    path_data = path.joinpath('Data')

    prev_app = pd.read_csv(path_data/"previous_application.csv")

    # CLEANING ANOMALIES
    # removing lines where both amt_applications and amt_credit are zero or null
    prev_app = prev_app.loc[~(
                (prev_app['AMT_APPLICATION'] == 0) &
                ((prev_app['AMT_CREDIT'] == 0) | (prev_app['AMT_CREDIT'].isna()))
                )]
    # replacing zero amt_application with amt_credit
    prev_app['AMT_APPLICATION'].mask(prev_app['AMT_APPLICATION'] == 0,
                                     prev_app['AMT_CREDIT'],
                                     axis=0,
                                     inplace=True)
    df = prev_app[['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE',
                   'DAYS_TERMINATION']].copy()
    df.mask(df == 365243.0, np.nan, inplace=True)
    prev_app[
        ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
         'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']] = df

    # CALCULATING GENERAL STATISTICS
    # previous application history
    aggregations = {'SK_ID_PREV': 'count',
                    'AMT_APPLICATION': ['sum', 'std'],
                    'DAYS_DECISION': ['min', 'max'],
                    'DAYS_LAST_DUE': 'max',
                    'DAYS_TERMINATION': 'max'
                    }

    df0 = prev_app.groupby(['SK_ID_CURR']).agg(aggregations)
    df0.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in df0.columns.tolist()])
    df0 = df0.reset_index()

    # custom stats
    df0['PREV_APP_DURATION'] = (df0['PREV_DAYS_DECISION_MAX'] - df0['PREV_DAYS_DECISION_MIN'] + 1)
    df0['PREV_AMT_APP_OVER_DURATION'] = df0['PREV_AMT_APPLICATION_SUM'] / df0['PREV_APP_DURATION']
    
    # previous application status
    df1 = prev_app[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby(['SK_ID_CURR']).value_counts(normalize=True)
    df1 = df1.to_frame().reset_index()
    df1 = df1.pivot(columns='NAME_CONTRACT_STATUS', index='SK_ID_CURR', values=0)
    df1.fillna(0, inplace=True)
    df1.columns = pd.Index(('PREV_' + df1.columns.str.upper()).to_list())
    df1 = df1.reset_index()

    prev_agg = df0.merge(df1, on='SK_ID_CURR')

    # STATISTICS ON REJECTED AND ACCEPTED APPLICATIONS
    rejected = prev_app.loc[prev_app['NAME_CONTRACT_STATUS'] == 'Refused']
    approved = prev_app.loc[prev_app['NAME_CONTRACT_STATUS'] == 'Approved']

    # frequency and amount per contract type
    df_list = []
    agg = {
        'AMT_ANNUITY': ['mean', 'std'],
        'AMT_APPLICATION': ['sum', 'mean'],
        'AMT_GOODS_PRICE': ['sum', 'std'],
        'AMT_CREDIT': ['sum', 'std']
    }

    for df, name in zip([rejected, approved], ['REJECTED', 'APPROVED']):
        # general statistics per status
        df2 = df.groupby(['SK_ID_CURR']).agg(agg)
        df2.columns = pd.Index(['PREV_' + name + '_' + e[0] + "_" + e[1].upper() for e in df2.columns.tolist()])
        df2['PREV_' + name + '_GOODS_APP_RATIO'] = df2['PREV_' + name + '_AMT_GOODS_PRICE_SUM'] / df2[
            'PREV_' + name + '_AMT_APPLICATION_SUM']
        df2['PREV_' + name + '_APP_ANNUITY_RATIO'] = df2['PREV_' + name + '_AMT_ANNUITY_MEAN'] / df2[
            'PREV_' + name + '_AMT_APPLICATION_MEAN']
        df2['PREV_' + name + '_CREDIT_APP_RATIO'] = df2['PREV_' + name + '_AMT_CREDIT_SUM'] / df2[
            'PREV_' + name + '_AMT_APPLICATION_SUM']
        df2 = df2.reset_index()
        # frequency per contract type
        df3 = df[['SK_ID_CURR', 'NAME_CONTRACT_TYPE']].groupby(['SK_ID_CURR']).value_counts(normalize=True)
        df3 = df3.to_frame().reset_index()
        df3 = df3.pivot(columns='NAME_CONTRACT_TYPE', index='SK_ID_CURR', values=0).reset_index()
        df3.fillna(0, inplace=True)
        df3.columns = pd.Index(['SK_ID_CURR'] + ('PREV_' + name + '_' + df3.columns.str.upper()[1:]).to_list())
        # amount per contract type
        df4 = rejected[['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_APPLICATION']].groupby(
            ['SK_ID_CURR', 'NAME_CONTRACT_TYPE']).sum().reset_index()
        df4 = df4.pivot(columns='NAME_CONTRACT_TYPE', values='AMT_APPLICATION', index='SK_ID_CURR').reset_index()
        df4.fillna(0, inplace=True)
        df4.columns = pd.Index(['SK_ID_CURR'] + ('PREV_' + name + '_AMT_APP_' + df4.columns.str.upper()[1:]).to_list())

        df_list.append(df2.merge(df3, on='SK_ID_CURR').merge(df4, on='SK_ID_CURR'))

    rej_agg = df_list[0]
    accept_agg = df_list[1]

    # adding rejection reason
    df5 = rejected[['SK_ID_CURR', 'CODE_REJECT_REASON']].groupby(['SK_ID_CURR']).value_counts()
    df5 = df5.to_frame().reset_index()
    df5 = df5.pivot(columns='CODE_REJECT_REASON', index='SK_ID_CURR', values=0)
    df5.columns = pd.Index(('PREV_REJECTED_REASON_' + df5.columns.str.upper()).to_list())
    df5.fillna(0, inplace=True)
    df5 = df5.reset_index()

    rej_agg = rej_agg.merge(df5, on='SK_ID_CURR')

    # BUILDING FINAL DATAFRAME
    prev_agg = prev_agg.merge(rej_agg, how='left', on='SK_ID_CURR')
    prev_agg = prev_agg.merge(accept_agg, how='left', on='SK_ID_CURR')

    # dropping non important columns
    prev_agg.drop(columns=['PREV_REJECTED_REASON_SYSTEM', 'PREV_REJECTED_REASON_XAP',
                           'PREV_REJECTED_REASON_XNA', 'PREV_REJECTED_REASON_VERIF'], inplace=True)

    return prev_agg
