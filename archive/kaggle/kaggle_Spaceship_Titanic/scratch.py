import pandas as pd

df = pd.read_csv("mlp_pred.csv")
df["Transported"] = (df["Transported_Prob"] > 0.5)  # 这一步生成的就是 bool
df.drop(columns=["Transported_Prob"], inplace=True)#inplace=True 表示在原数据上修改
df.to_csv("mlp_pred_bool.csv", index=False)
