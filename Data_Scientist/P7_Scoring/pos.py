import pandas as pd


def pos_cash():
    """ perform preprocessing of pos cash balance file """

    pos = pd.read_csv("./Data/POS_CASH_balance.csv")

    # STATISTICS ON DPD
    agg0 = {'SK_DPD': ['sum', 'mean'],
            'SK_DPD_DEF': ['sum', 'mean']
            }
    df0 = pos.groupby(['SK_ID_CURR']).agg(agg0)
    df0.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in df0.columns.tolist()])
    df0 = df0.reset_index()

    # STATISTICS ON INSTALMENTS

    # most recent month balance per previous loan
    df1 = pos.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']).groupby('SK_ID_PREV').tail(1)

    # percentage of completion on each previous loan
    df1['INSTALMENT_LEFT_RATIO'] = df1['CNT_INSTALMENT_FUTURE'] / df1['CNT_INSTALMENT']
    # stats per current application
    agg1 = {'CNT_INSTALMENT': ['mean'],  # mean loan duration
            'CNT_INSTALMENT_FUTURE': 'sum',  # total months left on instalments
            'INSTALMENT_LEFT_RATIO': 'mean',  # mean completion rate
            }
    df1 = df1.groupby(['SK_ID_CURR']).agg(agg1)
    df1.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in df1.columns.tolist()])
    df1 = df1.reset_index()

    # BUILDING FINAL DATAFRAME
    pos_agg = df0.merge(df1, on='SK_ID_CURR')

    return pos_agg
