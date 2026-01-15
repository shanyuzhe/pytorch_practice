import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "dataset", "train.csv") 

def preprocess():
    print(">>> 1. 读取数据...")
    df_train = pd.read_csv(data_path)
    df_test = pd.read_csv(data_path.replace('train.csv', 'test.csv'))
    len_train = len(df_train)
    
    # 合并数据
    df_all = pd.concat([df_train.drop('Transported', axis=1), df_test], axis=0, ignore_index=True)

    print(">>> 2. 特征工程...")
    
    # --- A. 基础特征 ---
    df_all['GroupId'] = df_all['PassengerId'].apply(lambda x: x.split('_')[0])
    group_counts = df_all['GroupId'].value_counts()
    df_all['GroupSize'] = df_all['GroupId'].map(group_counts)
    # 衍生特征：是否独自一人
    df_all['IsAlone'] = (df_all['GroupSize'] == 1).astype(int)
    
    # --- B. 智能填补分类特征 (重要优化) ---
    # HomePlanet, Destination, VIP 如果有缺失，用众数(出现最多的)填充
    # 否则 get_dummies 会把 NaN 变成全 0，丢失信息
    for col in ['HomePlanet', 'Destination', 'VIP', 'CryoSleep']:
        df_all[col] = df_all[col].fillna(df_all[col].mode()[0])

    # --- C. 船舱处理 (Cabin) ---
    # 先填补 Cabin 的 NaN，防止 split 报错
    df_all['Cabin'] = df_all['Cabin'].fillna('T/0/P') 
    df_all[['Deck', 'Num', 'Side']] = df_all['Cabin'].str.split('/', expand=True)
    
    # Deck 映射 (T 是这一层唯一的，样本极少，归为 F 或其他)
    df_all['Deck'] = df_all['Deck'].replace('T', 'F')
    
    # Num 转数字
    df_all['Num'] = pd.to_numeric(df_all['Num'], errors='coerce').fillna(-1)

    # --- D. 消费数据智能处理 ---
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df_all[spend_cols] = df_all[spend_cols].fillna(0)
    df_all['TotalSpend'] = df_all[spend_cols].sum(axis=1)
    df_all['HasSpent'] = (df_all['TotalSpend'] > 0).astype(int)

    # 逻辑修正：如果花了钱，绝对没休眠
    df_all.loc[df_all['TotalSpend'] > 0, 'CryoSleep'] = False
    df_all['CryoSleep'] = df_all['CryoSleep'].astype(int) # 转数字

    # --- E. 丢弃无用列 ---
    df_all.drop(['PassengerId', 'GroupId', 'Name', 'Cabin'], axis=1, inplace=True)

    print(">>> 3. 数值变换与归一化...")
    
    # 填补 Age
    df_all['Age'] = df_all['Age'].fillna(df_all['Age'].median())

    # 对数变换 (Log Transform) - 解决贫富差距问题
    for col in ['TotalSpend'] + spend_cols:
        df_all[col] = np.log1p(df_all[col])
    
    # 标准化 (StandardScaler)
    num_cols = ['Age', 'GroupSize', 'Num', 'TotalSpend'] + spend_cols
    scaler = StandardScaler()
    df_all[num_cols] = scaler.fit_transform(df_all[num_cols])

    print(">>> 4. One-Hot 编码...")
    cate_cols = df_all.select_dtypes(include=['object']).columns
    df_all = pd.get_dummies(df_all, columns=cate_cols)
    
    # 转换为 float32
    df_all = df_all.astype('float32')

    # --- 还原拆分 ---
    df_train_processed = df_all.iloc[:len_train].copy()
    df_test_processed = df_all.iloc[len_train:].copy()
    
    # 还原 Target 和 ID
    target = df_train['Transported'].map({True: 1, False: 0}).values
    df_train_processed['Transported'] = target
    
    df_train_processed['PassengerId'] = df_train['PassengerId'].values
    df_test_processed['PassengerId'] = df_test['PassengerId'].values
    
    print(f"特征工程完成。特征维度: {df_train_processed.shape[1]}")
    return df_train_processed, df_test_processed

if __name__ == '__main__':
    train, test = preprocess()
    # 检查一下是否有 NaN
    assert train.isnull().sum().sum() == 0, "训练集仍有 NaN"
    train.to_csv('dataset/train_processed.csv', index=False)
    test.to_csv('dataset/test_processed.csv', index=False)