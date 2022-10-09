import pandas as pd
from pathlib import Path

def bureau_and_balance():

    """ perform preprocessing of bureau and bureau balance files """
    path = Path(__file__).parent
    path_data = path.joinpath('Data')

    bureau_balance = pd.read_csv(path_data/"bureau_balance.csv")
    bureau = pd.read_csv(path_data/"bureau.csv")

    # bureau_balance aggregation before merge
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'count']}
    bb1 = bureau_balance[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby(['SK_ID_BUREAU']).agg(
        bb_aggregations).reset_index()
    bb1.columns = pd.Index(['SK_ID_BUREAU']
                           + ['MONTHS_BALANCE' + "_" + e[1].upper() for e in bb1.columns.tolist()[1:]])
    bb2 = bureau_balance.groupby(['SK_ID_BUREAU', 'STATUS']).count().reset_index()
    bb2 = bb2.pivot_table(values=['MONTHS_BALANCE'], index='SK_ID_BUREAU', columns='STATUS').reset_index()
    bb2.columns = pd.Index(['SK_ID_BUREAU']
                           + ['STATUS' + "_" + e[1].upper() for e in bb2.columns.tolist()[1:]])
    bb2.fillna(0, inplace=True)
    bb1 = bb1.merge(bb2, on='SK_ID_BUREAU')

    bureau = bureau.merge(bb1, how='left', on='SK_ID_BUREAU')

    # statistics on bureau
    bureau_aggregations = {
        'SK_ID_BUREAU': ['count'],
        'CREDIT_CURRENCY': ['nunique'],
        'DAYS_CREDIT': ['max'],
        'CREDIT_DAY_OVERDUE': ['max'],
        'DAYS_CREDIT_ENDDATE': ['max'],
        'AMT_CREDIT_MAX_OVERDUE': ['max'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_CREDIT_SUM': ['sum'],
        'AMT_CREDIT_SUM_DEBT': ['sum'],
        'AMT_CREDIT_SUM_LIMIT': ['mean'],
        'AMT_CREDIT_SUM_OVERDUE': ['sum'],
        # 'CREDIT_TYPE':['nunique','max'],
        'DAYS_CREDIT_UPDATE': ['max'],
        'AMT_ANNUITY': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_COUNT': ['sum'],
        'STATUS_0': ['sum'],
        'STATUS_1': ['sum'],
        'STATUS_2': ['sum'],
        'STATUS_3': ['sum'],
        'STATUS_4': ['sum'],
        'STATUS_5': ['sum'],
        'STATUS_C': ['sum'],
        'STATUS_X': ['sum']
    }
    bureau0 = bureau.groupby('SK_ID_CURR').agg(bureau_aggregations).reset_index()
    bureau0.columns = pd.Index(['SK_ID_CURR']
                               + ["BUREAU" + "_" + e[0] + "_" + e[1].upper() for e in bureau0.columns.tolist()[1:]])
    bureau1 = pd.crosstab(bureau['SK_ID_CURR'], bureau['CREDIT_ACTIVE'], normalize=0).reset_index()
    bureau1.columns = pd.Index(['SK_ID_CURR']
                               + ["BUREAU" + "_" + 'CREDIT' + "_" + e.upper() for e in bureau1.columns.tolist()[1:]])
    bureau = bureau1.merge(bureau0, on='SK_ID_CURR')

    return bureau
