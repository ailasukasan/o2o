import numpy as np
import pandas as pd
from datetime import date
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
off_train_data = pd.read_csv('C:/Users/86186/Desktop/phmnew/ccf_offline_stage1_train.csv',encoding='utf-8')
off_test_data = pd.read_csv('C:/Users/86186/Desktop/phmnew/ccf_offline_stage1_test_revised.csv',encoding='utf-8')
off_train_data = off_train_data.fillna('null')
off_test_data = off_test_data.fillna('null')

def label(row):
    if row['Date'] != 'null' and row['Date_received'] != 'null' and row['Coupon_id'] != 'null':
        td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
        return 1 if td <= pd.Timedelta(15, 'D') else 0
    if row['Date_received'] != 'null':
        return 0
    return -1

off_train_data['label'] = off_train_data.apply(label, axis=1)
off_train_data['label'].value_counts()

# 数据取样 
X = off_train_data.loc[:, off_train_data.columns != 'label']
y = off_train_data.loc[:, off_train_data.columns == 'label']
count_one_Class = len(off_train_data[off_train_data['label'] == 1])
one_Class_index = off_train_data[off_train_data['label'] == 1].index
zero_Class_index = off_train_data[off_train_data['label'] == 0].index
np.random.seed(25)
random_zero_index = np.random.choice(zero_Class_index, count_one_Class, replace=True)
sample = np.concatenate([one_Class_index, random_zero_index])
off_train_data = off_train_data.loc[sample, :]

print('label为1的数目：', len(one_Class_index))
print('label为0的数目：', len(zero_Class_index))
print('总数：', len(one_Class_index) + len(zero_Class_index))
print('抽样label为1的数目：', len(one_Class_index))
print('随机抽取label为0的数目：', len(random_zero_index))
print('抽样总数：', len(one_Class_index) + len(random_zero_index))
print('总样本形状：', off_train_data.shape)

# 第三部分 数据探索  
off_train_data['Distance'] = off_train_data['Distance'].replace('null', -1).astype(int)
off_test_data['Distance'] = off_test_data['Distance'].replace('null', -1).astype(int)
print('查看缺失值结果：\n', off_train_data.isnull().sum())

description = off_train_data.describe()
description.loc['range'] = description.loc['max'] - description.loc['min']
description.loc['var'] = description.loc['std'] ** 2
description.loc['dis'] = description.loc['75%'] - description.loc['25%']
print('描述性统计结果：\n', np.round(description, 2))

corr = off_train_data.corr(method='pearson')
print('相关系数矩阵为：\n', np.round(corr, 2))

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, vmax=1, square=True, cmap="Blues")
plt.title('相关性热力图')
plt.show()

D1 = np.array(off_train_data['Distance'].values)
D2 = np.array(off_test_data['Distance'].values)
plt.boxplot([D1, D2], labels=('off_train_data', 'off_test_data'))
plt.title('距离箱型图')
plt.show()

#    数据预处理  
def convertRate(row):
    if row == 'null':
        return 1.0
    elif ':' in str(row):
        rows = row.split(':')
        return 1.0 - float(rows[1]) / float(rows[0])
    else:
        return float(row)

def getDiscountType(row):
    if row == 'null':
        return -1
    elif ':' in row:
        return 1
    else:
        return 0

def Man_Rate(row):
    if row == 'null':
        return 0
    elif ':' in str(row):
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0

def Jian_Rate(row):
    if row == 'null':
        return 0
    elif ':' in str(row):
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0

off_train_data['Dis_rate'] = off_train_data['Discount_rate'].apply(convertRate)
off_train_data['Discount_type'] = off_train_data['Discount_rate'].apply(getDiscountType)
off_train_data['Discount_man'] = off_train_data['Discount_rate'].apply(Man_Rate)
off_train_data['Discount_jian'] = off_train_data['Discount_rate'].apply(Jian_Rate)
off_test_data['Dis_rate'] = off_test_data['Discount_rate'].apply(convertRate)
off_test_data['Discount_type'] = off_test_data['Discount_rate'].apply(getDiscountType)
off_test_data['Discount_man'] = off_test_data['Discount_rate'].apply(Man_Rate)
off_test_data['Discount_jian'] = off_test_data['Discount_rate'].apply(Jian_Rate)

data = off_train_data[off_train_data['label'] != -1]
data = data.fillna(-1)
data['label'].value_counts()

def getWeekday(row):
    return date(int(row[:4]), int(row[4:6]), int(row[6:8])).weekday() + 1 if row != 'null' else row

data['Weekday'] = data['Date_received'].astype(str).apply(getWeekday)
off_test_data['Weekday'] = off_test_data['Date_received'].astype(str).apply(getWeekday)
data['Is_weekend'] = data['Weekday'].apply(lambda x: 1 if x in [6, 7] else 0)
off_test_data['Is_weekend'] = off_test_data['Weekday'].apply(lambda x: 1 if x in [6, 7] else 0)

def One_hot(df):
    weekdaycols = ['weekday' + str(i) for i in range(1, 8)]
    tmpdf = pd.get_dummies(df['Weekday'].replace('null', np.nan))
    tmpdf.columns = weekdaycols
    df[weekdaycols] = tmpdf
    return df

data = One_hot(data)
off_test_data = One_hot(off_test_data)

def func(data):
    f = data[['User_id', 'Coupon_id']].copy()
    f['rec_coupon'] = 1
    f = f.groupby('User_id').agg('sum').reset_index()
 
    f1 = data[['Coupon_id']].copy()
    l1 = len(f1)
    f1['Number_coupon'] = 1
    f1 = f1.groupby('Coupon_id').agg('sum').reset_index()
    f1['Coupon_popu'] = f1['Number_coupon'] / l1
 
    f2 = data[['User_id', 'Merchant_id']].copy()
    l2 = len(f2)
    f2['Number_merchant'] = 1
    f2 = f2.groupby('Merchant_id').agg('sum').reset_index()
    f2['Merchant_popu'] = f2['Number_merchant'] / l2
 
    d0 = pd.merge(data, f[['User_id', 'rec_coupon']], on='User_id')
    d1 = pd.merge(d0, f1[['Coupon_id', 'Coupon_popu']], on='Coupon_id')
    d2 = pd.merge(d1, f2[['Merchant_id', 'Merchant_popu']], on='Merchant_id')
    return d2

new_data = func(data)
new_test_data = func(off_test_data)

def Get_mer_dis(new_data):
    new_data['Distance'].value_counts()
    md1 = new_data[new_data.Coupon_id != 'null'][['Merchant_id', 'Distance']]
    md1.replace('null', -1, inplace=True)
    md1.replace(-1, np.nan, inplace=True)
    merchant_feature = md1.groupby('Merchant_id')['Distance'].agg(
        merchant_min_distance='min',
        merchant_max_distance='max',
        merchant_mean_distance='mean',
        merchant_median_distance='median'
    ).reset_index()
    new_data = pd.merge(new_data, merchant_feature, on='Merchant_id', how='left')
    return new_data

new_data = Get_mer_dis(new_data)
new_test_data = Get_mer_dis(new_test_data)

x = np.arange(-1, 11)
dis1 = new_data['Distance'].value_counts().sort_index().values
dis2 = new_test_data['Distance'].value_counts().sort_index().values
plt.bar(x, dis1, tick_label=x, label='off_train_data', width=0.3)
plt.bar(x + 0.3, dis2, label='off_test_data', width=0.3)
plt.xlabel('距离')
plt.ylabel('计数')
plt.title('距离计数分布情况')
plt.legend()
plt.show()

def get_distance_type(row):
    if row == -1:
        return -1
    elif row <= 2:
        return 0
    elif row <= 5:
        return 1
    elif row <= 9:
        return 2
    else:
        return 3

new_data['Distance_type'] = new_data['Distance'].apply(get_distance_type)
new_test_data['Distance_type'] = new_test_data['Distance'].apply(get_distance_type)

x1 = np.arange(-1, 4)
dis_type1 = new_data['Distance_type'].value_counts().sort_index().values
dis_type2 = new_test_data['Distance_type'].value_counts().sort_index().values
plt.bar(x1, dis_type1, tick_label=x1, label='off_train_data', width=0.3)
plt.bar(x1 + 0.3, dis_type2, label='off_test_data', width=0.3)
plt.xlabel('距离类型')
plt.ylabel('计数')
plt.title('距离类型计数分布情况')
plt.legend()
plt.show()

def Get_dis_hot(df):
    discols = ['Distance' + str(i) for i in range(-1, 11)]
    tmpdf = pd.get_dummies(df['Distance'].replace('null', np.nan))
    tmpdf.columns = discols
    df[discols] = tmpdf
    return df

new_data = Get_dis_hot(new_data)
new_test_data = Get_dis_hot(new_test_data)
new_data = new_data.fillna(-1)
new_test_data = new_test_data.fillna(-1)
new_data.isnull().sum()
new_test_data.isnull().sum()

#   挖掘建模 
new_data.columns
new_test_data.columns

features = ['Dis_rate', 'Discount_type', 'Discount_man', 'Discount_jian',
            'Distance', 'Distance_type', 'Distance-1', 'Distance0',
            'Distance1', 'Distance2', 'Distance3', 'Distance4', 'Distance5',
            'Distance6', 'Distance7', 'Distance8', 'Distance9', 'Distance10',
            'rec_coupon', 'Coupon_popu', 'Merchant_popu', 'merchant_min_distance',
            'merchant_max_distance', 'merchant_mean_distance', 'merchant_median_distance',
            'Is_weekend', 'Weekday', 'weekday1', 'weekday2', 'weekday3', 'weekday4',
            'weekday5', 'weekday6', 'weekday7']

X = new_data[features]
y = new_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

GBC_model = GradientBoostingClassifier(n_estimators=160, max_depth=4, learning_rate=0.13)
GBC_model.fit(X, y)
y_predict = GBC_model.predict_proba(X_test)[:, 1]
y_auc = roc_auc_score(y_test, y_predict)

# 模型评价 
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, 'r')
axline = np.array([0., 0.2, 0.4, 0.6, 0.8, 1.0])
plt.plot(axline, axline, 'gray', linestyle='--', alpha=0.5)
plt.grid(b=True, axis='both', alpha=0.3)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('AUC = %0.2f' % roc_auc)
plt.show()

#  模型应用  
pre_test = new_test_data[features]
result = GBC_model.predict_proba(pre_test)[:, 1]
test_result = new_test_data[['User_id', 'Coupon_id', 'Date_received']]
test_result['Probability'] = result
test_result['Probability'].describe()
test_result.to_csv('./new_sample_submission.csv', index=None, header=None)
