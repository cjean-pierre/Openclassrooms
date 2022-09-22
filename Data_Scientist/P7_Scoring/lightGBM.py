import numpy as np
import pandas as pd
import gc
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, recall_score, \
    precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


def kfold_lightgbm(df, num_folds, debug=False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()].copy()
    test_df = df[df['TARGET'].isnull()].copy()
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df

    train_df.replace(np.inf, np.nan, inplace=True)

    # Cross validation model
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    a_preds = np.zeros(train_df.shape[0])

    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU',
                                                      'SK_ID_PREV', 'index']]
    contrib = [feats+['exp_value']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # SMOTE resampling
        over = SMOTE(sampling_strategy=0.5)
        under = RandomUnderSampler(sampling_strategy=0.9)
        steps = [('over', over), ('under', under)]
        pipeline = Pipeline(steps=steps)

        x_resampled, y_resampled = pipeline.fit_resample(train_x, train_y)

        # LightGBM parameters
        clf = LGBMClassifier(
            objective='binary',
            n_estimators=5000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            # subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            class_weight='balanced'
            # bagging_freq=200,
            # pos_bagging=0.1,
            # neg_bagging=0.1
        )

        clf.fit(x_resampled, y_resampled, eval_set=[(x_resampled, y_resampled), (valid_x, valid_y)],
                # eval_class_weight=[{0: 12}, {1: 1}],
                eval_metric='auc',
                callbacks=[log_evaluation(200), early_stopping(200)])

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        contrib.append(clf.predict_proba(valid_x, num_iteration=clf.best_iteration_, pred_contrib=True))
        
        # Attention ici !
        a_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration_)
        
        
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        # print('Fold %2d balanced_accuracy : %.6f' % (n_fold + 1, balanced_accuracy_score(valid_y, a_preds[valid_idx])))
        # print('Fold %2d accuracy : %.6f' % (n_fold + 1, accuracy_score(valid_y, a_preds[valid_idx])))

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    print('Full balanced accuracy score %.6f' % balanced_accuracy_score(train_df['TARGET'], a_preds))
    # print('Full accuracy score %.6f' % accuracy_score(train_df['TARGET'], a_preds))
    # print('Full recall score %.6f' % recall_score(train_df['TARGET'], a_preds))
    print('Full precision_recall_fscore: ', precision_recall_fscore_support(train_df['TARGET'], a_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df.loc[:, 'TARGET'] = sub_preds
        test_predictions = test_df[['SK_ID_CURR', 'TARGET']].copy()

    display_importances(feature_importance_df)
    # write feature contribution file
    feat_contrib = pd.DataFrame(np.vstack(contrib))
    train_preds = pd.concat([train_df['TARGET'], pd.Series(oof_preds), pd.Series(a_preds)], axis=1)

    return feature_importance_df, test_predictions, feat_contrib, train_preds


# Display/plot feature importance


def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean()
    cols = cols.sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')
