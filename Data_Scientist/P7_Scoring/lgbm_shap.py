from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
import shap
import numpy as np
import matplotlib.colors


def lgbm_shap(train_df, test_df, contrib=True):

    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU',
                                                      'SK_ID_PREV', 'index']]
    train_x, valid_x, train_y, valid_y = train_test_split(train_df[feats], train_df['TARGET'],
                                                          stratify=train_df['TARGET']
                                                          )

    # LightGBM parameters
    params = {
        'objective': 'binary',
        'n_estimators': 5000,
        'learning_rate': 0.02,
        'num_leaves': 34,
        'colsample_bytree': 0.9497036,
        'max_depth': 8,
        'reg_alpha': 0.041545473,
        'reg_lambda': 0.0735294,
        'min_split_gain': 0.0222415,
        'min_child_weight': 39.3259775,
        'class_weight': {0: 1, 1: 6}
        }
    clf = LGBMClassifier()
    clf.set_params(**params)

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            # eval_class_weight=[{0: 12}, {1: 1}],
            eval_metric='auc',
            callbacks=[log_evaluation(200), early_stopping(200)])
    sub_preds = clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1]

    if contrib:
        contribs = clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_, pred_contrib=True)

        return sub_preds, contribs, feats

    else:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(test_df[feats])
        expected_values = explainer.expected_value
        shap_values_bar = explainer(test_df[feats])

        return sub_preds, shap_values, expected_values, shap_values_bar, feats


def shap_viz_prep(contribs, feats, test_df):
    """prepare data for shap vizualisations"""
    shap_values = contribs[:, :-1]
    exp_values = contribs[:, -2:-1]
    feat_names = [feat.capitalize() for feat in feats]
    feat_values = np.array(test_df[feats])

    return shap_values, exp_values, feat_values, feat_names


def plot_shap_summary(shap_values, feat_values, feat_names):
    hl_colors = ['#CADAE6', '#A6C1CF', '#86A6B0', '#546E7A', '#37474F']
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('hl_cmap', colors=hl_colors, N=100)

    fig_summary = shap.summary_plot(shap_values, feat_values, feature_names=feat_names, max_display=20,
                                    cmap=cmap2,
                                    plot_size=0.35,
                                    show=False)
    return fig_summary
