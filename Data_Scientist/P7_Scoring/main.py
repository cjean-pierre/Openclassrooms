from preprocess import *
from lightGBM import*


def main():

    with timer("Run preprocessing"):
        train_df, test_df = preprocessing()

    with timer("Run LightGBM with kfold"):

        feat_importance, oof_preds, _ = kfold_lightgbm(train_df, test_df, num_folds=5,
                                                       smote=False, class_weight={0: 1, 1: 6},
                                                       vis=True, contrib=False)

        score_lightgbm(oof_preds, train_df['TARGET'], threshold=0.3, beta=1, verbose=True)

        display_importances(feat_importance, save_files=False)


with timer("Full model run"):
    main()

if __name__ == '__main__':
    print('I love coding')
