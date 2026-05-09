import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 加载数据
data = pd.read_csv('H:/peptide/batchNorm/all_data_split2/processed_permeability_data_clipped_modified.csv')

# 检查数据范围
value_min = data['Standardized_Value'].min()
value_max = data['Standardized_Value'].max()
print(f"Standardized_Value范围: {value_min} 到 {value_max}")

# 数值在-4到-8之间，选择8个分箱，每个分箱宽度为0.5
n_bins = 8
bins = np.linspace(value_min, value_max, n_bins + 1)
data['bin'] = pd.qcut(data['Standardized_Value'], q=n_bins, labels=False, duplicates='drop')

bin_stats = data.groupby('bin')['Standardized_Value'].agg(['count', 'min', 'max']).reset_index()
print("\n各个分箱的统计信息:")
print(bin_stats)

# 创建5个fold
n_folds = 5
data['fold'] = -1  # 初始化fold列

# 对每个分箱进行处理，确保样本均匀分布到5个fold
for bin_idx in range(n_bins):
    bin_indices = data[data['bin'] == bin_idx].index.tolist()
    
    if bin_indices:
        # 随机打乱索引
        np.random.seed(42)  # 设置随机种子以确保可重复性
        np.random.shuffle(bin_indices)
        
        # 将索引均匀分配到5个fold
        for i, idx in enumerate(bin_indices):
            data.loc[idx, 'fold'] = i % n_folds

# 生成5个fold的训练集和测试集
for fold in range(n_folds):
    train_data = data[data['fold'] != fold].drop(columns=['bin', 'fold'])
    test_data = data[data['fold'] == fold].drop(columns=['bin', 'fold'])
    
    # 保存训练集和测试集
    train_file = f'fold_{fold+1}_train.csv'
    test_file = f'fold_{fold+1}_test.csv'
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    print(f"Fold {fold+1}:")
    print(f"  训练集: {len(train_data)} 样本, 保存为 {train_file}")
    print(f"  测试集: {len(test_data)} 样本, 保存为 {test_file}")
    print(f"  训练集:测试集比例 = {len(train_data)/len(test_data):.2f}:1")

# 检查每个分箱在各个fold中的分布情况
bin_fold_dist = pd.crosstab(data['bin'], data['fold'])
print("\n各个分箱在各个fold中的分布情况:")
print(bin_fold_dist)