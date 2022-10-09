import pandas as pd
from pathlib import Path

def credit_card_balance():
    """ perform preprocessing of credit card balance """
    path = Path(__file__).parent
    path_data = path.joinpath('Data')

    ccb = pd.read_csv(path_data/"credit_card_balance.csv")

    # statistics for 12 months
    ccb6 = ccb.groupby(['SK_ID_CURR', 'SK_ID_PREV']).tail(12)

    ccb6_1 = ccb6[['SK_ID_CURR', 'AMT_DRAWINGS_ATM_CURRENT',
                   'AMT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_ATM_CURRENT',
                   'CNT_DRAWINGS_CURRENT', 'SK_DPD']].groupby('SK_ID_CURR').sum()
    ccb6_2 = ccb6[['SK_ID_CURR', 'SK_ID_PREV',
                   'AMT_BALANCE']].groupby(['SK_ID_CURR',
                                            'SK_ID_PREV']).mean().groupby('SK_ID_CURR').sum()
    ccb_agg = ccb6_2.merge(ccb6_1, on='SK_ID_CURR')

    ccb_agg.columns = pd.Index(
        ['CCB_AMT_BALANCE_MEAN_12M']
        + ('CCB_' + ccb_agg.columns.str.upper()[1:] + '_SUM_12M').to_list())

    return ccb_agg
