import time
from datetime import timedelta
from contextlib import contextmanager
import re

from prev_app import *
from pos import *
from bureau import*
from credit_card import*
from installments import *
from application import*


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    t_delta = str(timedelta(seconds=time.time()-t0)).split(':')
    print("{} - done in {}h {}min {}sec".format(title, t_delta[0], t_delta[1], t_delta[2]))


def preprocessing():
    """ perform preprocessing and return train and test dataframes"""

    with timer("Process application files"):
        df = application_train_test()
        print("Application df shape:", df.shape)

    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance()
        print("Bureau df shape:", bureau.shape)
        df = df.merge(bureau, how='left', on='SK_ID_CURR')
        del bureau

    with timer("Process previous_applications"):
        prev = previous_applications()
        print("Previous applications df shape:", prev.shape)
        df = df.merge(prev, how='left', on='SK_ID_CURR')
        del prev

    with timer("Process POS-CASH balance"):
        pos = pos_cash()
        print("Pos-cash balance df shape:", pos.shape)
        df = df.merge(pos, how='left', on='SK_ID_CURR')
        del pos

    with timer("Process installments payments"):
        ins = installments_payments()
        print("Installments payments df shape:", ins.shape)
        df = df.merge(ins, how='left', on='SK_ID_CURR')
        del ins

    with timer("Process credit card balance"):
        cc = credit_card_balance()
        print("Credit card balance df shape:", cc.shape)
        df = df.merge(cc, how='left', on='SK_ID_CURR')
        del cc

    df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x), inplace=True)
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()].copy()
    train_df.to_csv('train_df.csv')
    test_df = df[df['TARGET'].isnull()].copy()
    print("Preprocessing Done. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df

    return train_df, test_df
