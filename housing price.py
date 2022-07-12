from warnings import simplefilter
import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

simplefilter(action='ignore', category=FutureWarning)


# 自定义转换器，用于处理Dataframe
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# fetch_housing_data()
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# def split_train_test(data, test_ratio=0.2):
#     np.random.seed(42)
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]

df = load_housing_data()
# print(df)
# print(df['ocean_proximity'].value_counts())
# df.hist(bins=50,figsize=(20,15))
# plt.show()

# 数据量较大时，无需进行分层采样
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
# print(train_set)

# 数据量较小时，需要保持重要的属性的比例一致，采用StratfiedShuffleSplit
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

housing = train_set.copy()
# 地理数据可视化,alpha透明度（点多的就更明显，用来表示稀疏度），s表示圈大小（人口），c表示颜色（价格）
# housing.plot(kind='scatter', x='longitude',y='latitude',alpha=0.4,s=housing["population"]/100, label="population",
#     c="median_house_value", cmap=plt.get_cmap("jet"))
# plt.legend()
# plt.show()

housing = train_set.drop('median_house_value', axis=1)
housing_labels = train_set['median_house_value'].copy()

# 数据清洗

# 采用dropna，fillna这些方法
# housing.dropna(subset=["total_bedrooms"])    # 选项1
# housing.drop("total_bedrooms", axis=1)       # 选项2
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median)     # 选项3
# 采用imputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)  ##去掉文本属性的数据副本
imputer.fit(housing_num)
X = imputer.transform(housing_num)
# print(X)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
# print(housing_tr)

# 处理文本和类别属性
encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
# 单个文本特征
# housing_cat_encoded=encoder.fit_transform(housing_cat)
# print(housing_cat_encoded)
# 多个文本特征
# housing_cat_encoded, housing_categories = housing_cat.factorize()
# print(housing_cat_encoded,housing_categories)
# 采用独热编码（一个n*1的数组，给每个分类创建二元属性，只有一个属性为1（热），其余都为0（冷））
# encoder=OneHotEncoder()
# housing_cat_1hot=encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
# 使用LabelBinarizer一步执行从文本分类到整数分类再到独热向量
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot = pd.DataFrame(housing_cat_1hot)

# 特征缩放，标准化（不像归一化限定值到某个范围）受到异常值的影响较小
sc = StandardScaler()
X = sc.fit(housing_num)
housing_sc = X.transform(housing_num)
housing_sc = pd.DataFrame(housing_sc, columns=housing_num.columns)
# print(housing_sc)

# Pipeline转换流水线，将处理数值和文本属性、标准化等等放到流水线上
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', LabelBinarizer()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])
# housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)

housing_prepared = pd.DataFrame()
housing_prepared = pd.concat([housing_prepared, housing_sc], axis=1)
housing_prepared = pd.concat([housing_prepared, housing_cat_1hot], axis=1)
# print(housing_prepared)

# 线性回归
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
# some_data=housing.iloc[:5]
# some_labels=housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)
# print(some_data_prepared)

# 计算RMSE(均方误差根）
housing_preditions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_preditions)
lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)  #68433.94,大多数房价中位数在120000美元到150000美元之间，误差68433.94美元比较差

# 训练决策树模型
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
# tree_preditions=tree_reg.predict(housing_prepared)
# tree_mse=mean_squared_error(housing_labels,tree_preditions)
# tree_rmse=np.sqrt(tree_mse)
# print(tree_rmse) #0.0,模型严重过拟合数据

# 交叉验证
# 它随机地将训练集分成十个不同的子集，成为“折”，然后训练评估决策树模型 10 次，每次选一个不用的折来做评估，用其它 9 个来做训练。
# scores=cross_val_score(tree_reg,housing_prepared,housing_labels,scoring='neg_mean_squared_error',cv=10)
# tree_rmse_scores=np.sqrt(-scores)
# print(tree_rmse_scores) #[67364.94846009 70980.15190066 67314.11221017 70863.19616589
# 67838.50645646 67731.39706099 63387.21291707 71158.19613597
# 69015.94497194 68670.33314904] 交叉验证能得到模型性能的评估好坏,决策树过拟合严重，比线性模型还差
# lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores=np.sqrt(-lin_scores)
# print(lin_rmse_scores)[65581.45520649 71711.35784404 68143.02388491 66855.55244479
#  69440.38017435 65640.36503235 65861.37192245 69898.33048393
#  73117.94692191 69704.17693297]

# 随机森林模型
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
# forest_predictions=forest_reg.predict(housing_prepared)
# forest_mse=mean_squared_error(housing_labels,forest_predictions)
# forest_rmse=np.sqrt(forest_mse)
# print(forest_rmse) #18104.90093349288
# forest_scores=cross_val_score(forest_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores=np.sqrt(-forest_scores)
# print(forest_rmse_scores)[46629.07845037 51174.74043201 47489.75382816 50040.37385354
#  49845.40601323 46898.72242261 45636.52714793 50716.94798253
#  50253.05209442 49811.31405724]

# 支持向量机SVM
# clf=svm.SVC(C=1,kernel='rbf',gamma=0.01)
# clf.fit(housing_prepared,housing_labels)
# clf_predictions=clf.predict(housing_prepared)
# clf_mse=mean_squared_error(housing_labels,clf_predictions)
# clf_rmse=np.sqrt(clf_mse)
# print(clf_rmse)

# 保存和调用模型
# joblib.dump(forest_reg,'my_model.pkl')
# my_model_loaded=joblib.load('my_model.pkl')
# scores=cross_val_score(my_model_loaded, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
# rmse_scores=np.sqrt(-scores)
# print(rmse_scores)[46978.82790991 50818.49850513 47681.19327134 50543.45745238
#  50030.87700719 47365.4032604  45402.16550243 50766.94496366
#  50187.73258734 49900.05004044]

# # 模型微调——网格搜索，交叉验证所有可能的超参数值的组合
forest_reg2 = RandomForestRegressor()
param_grid1 = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
grid_search = GridSearchCV(forest_reg2, param_grid1, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
# # print(grid_search.best_params_){'max_features': 8, 'n_estimators': 30}
# # print(grid_search.best_estimator_)RandomForestRegressor(max_features=8, n_estimators=30)
# # forest_reg2=RandomForestRegressor(max_features=8, n_estimators=30)
# # forest_reg2.fit(housing_prepared,housing_labels)
# # joblib.dump(forest_reg2,'my_model.pkl')
#
# # 模型微调——随机搜索
forest_reg3 = RandomForestRegressor()
param_grid2 = {'n_estimators': list(range(10, 20)), 'max_features': list(range(2, 6))}
random_grid_search = RandomizedSearchCV(forest_reg3, param_grid2, cv=5, scoring='neg_mean_squared_error', n_iter=10,
                                        random_state=5)
random_grid_search.fit(housing_prepared, housing_labels)
# # print(random_grid_search.best_params_){'n_estimators': 19, 'max_features': 5}
# # print(random_grid_search.best_estimator_)RandomForestRegressor(max_features=5, n_estimators=19)
forest_reg3 = RandomForestRegressor(max_features=5, n_estimators=19)
forest_reg3.fit(housing_prepared, housing_labels)
joblib.dump(forest_reg3, 'my_model.pkl')
