import pandas as pd

# 读取两个模型的概率结果
mlp = pd.read_csv('mlp_pred.csv')
lgbm = pd.read_csv('lgbm_pred.csv')

# 简单的加权平均
# 通常 Tree 模型在这个比赛稍微强一点，可以给 0.6 的权重
# 你可以尝试 0.5 vs 0.5 或者 0.4 vs 0.6
final_prob = 0.3 * mlp['Transported_Prob'] + 0.7 * lgbm['Transported_Prob']

# 转成 True/False 提交
final_pred = (final_prob > 0.5).astype(bool)

submission = pd.DataFrame({
    'PassengerId': mlp['PassengerId'],
    'Transported': final_pred
})

submission.to_csv('submission_ensemble.csv', index=False)
print("融合完成！去提交 submission_ensemble.csv 吧！")