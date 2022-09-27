import numpy as np
import pandas as pd
import gc
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from yellowbrick.classifier import DiscriminationThreshold


def smote_resampling(x, y, overstrat=0.5, understrat=0.9):
    """
    perform over and under resampling using SMOTE
    Args
        x : dataframe with sample to be resampled
        y : corresponding targets to be resampled
    return
        x_resampled : resampled dataframe
        y_resampled : resampled target
    """
    x.replace(np.inf, np.nan, inplace=True)
    x.fillna(0, inplace=True)

    over = SMOTE(sampling_strategy=overstrat)
    under = RandomUnderSampler(sampling_strategy=understrat)
    steps = [('over', over), ('under', under)]
    pipeline = Pipeline(steps=steps)

    x_resampled, y_resampled = pipeline.fit_resample(x, y)

    return x_resampled, y_resampled


def discrimin_threshold(x, y, estimator, n_trials, outpath, is_fitted=True, fbeta=1.3):
    """
    fit and save discrimination threshold visualizer
    """
    visualizer = DiscriminationThreshold(estimator=estimator, n_trials=n_trials, is_fitted=is_fitted, fbeta=fbeta)
    visualizer.fit(x, y)
    visualizer.show(outpath=outpath, clear_figure=True)


def kfold_lightgbm(train_df, test_df, num_folds, smote=False, class_weight=None, vis=True, contrib=False):

    """
        perform stratified cross validation on lightgbm model
    """
    # Cross validation model
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])

    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU',
                                                      'SK_ID_PREV', 'index']]
    oof_contrib = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters

        params = {
            'objective': 'binary',
            'n_estimators': 5000,
            'learning_rate': 0.02,
            'num_leaves': 34,
            'colsample_bytree': 0.9497036,
            # 'subsample':0.8715623,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight': 39.3259775,
            # 'bagging_freq':200,
            # 'pos_bagging':0.1,
            # 'neg_bagging':0.1
        }
        clf = LGBMClassifier()

        # updating params according to class weight strategy  and fit classifier
        if smote:
            # SMOTE resampling
            print('SMOTE resampling', 'fold : ', n_fold + 1)
            params['class_weight'] = None
            clf.set_params(**params)
            x_resampled, y_resampled = smote_resampling(train_x, train_y)
            clf.fit(x_resampled, y_resampled, eval_set=[(x_resampled, y_resampled), (valid_x, valid_y)],
                    # eval_class_weight=[{0: 12}, {1: 1}],
                    eval_metric='auc',
                    callbacks=[log_evaluation(200), early_stopping(200)])
            if vis:
                discrimin_threshold(valid_x, valid_y, estimator=clf, n_trials=1,
                                    outpath=f'smote_discrim{n_fold}', is_fitted=True, fbeta=1.3)

        else:
            print('class_weight :', class_weight)
            params['class_weight'] = class_weight
            clf.set_params(**params)
            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                    # eval_class_weight=[{0: 12}, {1: 1}],
                    eval_metric='auc',
                    callbacks=[log_evaluation(200), early_stopping(200)])
            if vis:
                discrimin_threshold(valid_x, valid_y, estimator=clf, n_trials=1,
                                    outpath=f'discrim_cweight_{n_fold}', is_fitted=True, fbeta=1.3)

        # predictions

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        if contrib:
            oof_contrib.append(clf.predict_proba(valid_x, num_iteration=clf.best_iteration_, pred_contrib=True))

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    return feature_importance_df, oof_preds, sub_preds, np.vstack(oof_contrib)


def score_lightgbm(preds, target, threshold=0.5, beta=1, verbose=True):

    adj_preds = np.where(preds > threshold, 1, 0)

    roc_auc = roc_auc_score(target, preds)
    accuracy = accuracy_score(target, adj_preds)
    prfbeta = precision_recall_fscore_support(target, adj_preds, beta=beta)
    prfbeta_avg = precision_recall_fscore_support(target, adj_preds, beta=beta, average='weighted')

    prf_all = np.hstack((np.vstack(prfbeta), np.array(prfbeta_avg, ndmin=2).T))
    prf_summary = pd.DataFrame(prf_all, columns=['Class_0', 'Class_1', 'weighted_average'],
                               index=['precision', 'recall', f'f{beta}', 'population'])

    if verbose:
        print('Full AUC score %.6f' % roc_auc)
        print('Full accuracy score %.6f' % accuracy)
        print(f'Full precision_recall_f{beta} score: \n', prf_summary)

    return prf_summary


def display_importances(feature_importance_df_, save_files=True):

    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean()
    cols = cols.sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')
    if save_files:
        feature_importance_df_.to_csv('features_importance.csv')
