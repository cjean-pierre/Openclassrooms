import pandas as pd
from pathlib import Path


def installments_payments():
    """ perform preprocessing of credit installments_payments """
    path = Path(__file__).parent
    path_data = path.joinpath('Data')

    ins = pd.read_csv(path_data/"installments_payments.csv")

    # CLEANING
    # treating nan values
    mask = (ins['AMT_INSTALMENT'] == 0) & (ins['AMT_PAYMENT'].isna())
    ins.loc[mask, 'DAYS_ENTRY_PAYMENT'] = ins.loc[mask, 'DAYS_INSTALMENT']
    ins.fillna(value=0, inplace=True)

    # payments versus installments
    mask = ~(ins['AMT_INSTALMENT'] == 0)
    ins.loc[mask, 'PAYMENT_PERC'] = ins.loc[mask, 'AMT_PAYMENT'] / ins.loc[mask, 'AMT_INSTALMENT']
    ins.loc[~mask, 'PAYMENT_PERC'] = 1

    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'SK_ID_PREV': 'count',
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'NUM_INSTALMENT_NUMBER': ['size'],
        'DPD': ['max', 'mean', 'sum', 'std'],
        'DBD': ['max', 'mean', 'sum', 'std'],
        'PAYMENT_PERC': ['min', 'mean', 'std'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'std'],
        'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum', 'std'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'std'],

    }

    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

    return ins_agg
