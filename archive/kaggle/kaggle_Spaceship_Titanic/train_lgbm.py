import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os

def main():
    print(">>> 正在启动 LightGBM 训练...")
    
    # 1. 读取和你 MLP 一模一样的数据
    df_train = pd.read_csv('dataset/train_processed.csv')
    df_test = pd.read_csv('dataset/test_processed.csv')

    X = df_train.drop(['Transported', 'PassengerId'], axis=1).values
    y = df_train['Transported'].values
    X_test = df_test.drop(['PassengerId'], axis=1).values
    sub_ids = df_test['PassengerId']

    # 2. 设置 LightGBM 参数 (这是针对 Spaceship Titanic 调优过的参数)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,  # 学习率慢一点，效果更好
        'n_estimators': 2000,   # 树多一点
        'num_leaves': 31,
        'max_depth': 8,
        'min_child_samples': 20,
        'subsample': 0.7,       # 随机采样，防止过拟合
        'colsample_bytree': 0.7,# 特征采样
        'random_state': 42,
        'verbose': -1
    }

    # 3. 五折交叉验证 (K-Fold)
    # 这比单次 train_test_split 强在：它利用了 100% 的数据进行训练
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(X.shape[0])     # 验证集预测结果
    test_preds = np.zeros(X_test.shape[0]) # 测试集预测结果
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        
        # 训练
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )

        # 预测
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        
        # 累加测试集预测
        test_preds += model.predict_proba(X_test)[:, 1] / 10
        
        acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
        scores.append(acc)
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")

    print(f"\nLightGBM 平均准确率: {np.mean(scores):.4f}")
    
    # 保存 LightGBM 的预测结果 (概率值)
    # 我们不直接存 0/1，存概率值方便后面融合
    result_df = pd.DataFrame({'PassengerId': sub_ids, 'Transported_Prob': test_preds})
    result_df.to_csv('lgbm_pred.csv', index=False)
    print("LightGBM 预测结果已保存为 lgbm_pred.csv")

if __name__ == '__main__':
    main()