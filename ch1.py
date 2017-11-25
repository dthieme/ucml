import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

HOUSING_PATH = "datasets/housing"

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

df_housing = load_housing_data()
print(df_housing.head())


df_housing.hist(bins=50, figsize=(20,150))
#plt.show()

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indicies = shuffled_indices[test_set_size:]
    return data.iloc[train_indicies], data.iloc[test_indices]

train_set, test_set = split_train_test(df_housing, .2)
#print(len(train_set), "train + ", len(test_set), "test")

df_housing["income_cat"] = np.ceil(df_housing["median_income"] / 1.5)
df_housing["income_cat"].where(df_housing["income_cat"] < 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=.2,  random_state=42)
for train_index, test_index in split.split(df_housing, df_housing["income_cat"]):
    strat_train_set = df_housing.loc[train_index]
    strat_test_set = df_housing.loc[test_index]

df_housing["income_cat"].value_counts() / len(df_housing)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

df_housing = strat_train_set.copy()
#df_housing.plot(kind="scatter", x="longitude", y="latitude", alpha=.1)
#plt.show()
#df_housing.plot(kind="scatter",
#           x="longitude",
#           y="latitude",
#           alpha=.4,
#           s=df_housing["population"]/100,
#           label="population",
#           figsize=(10,7),
#           c="median_house_value",
#           cmap=plt.get_cmap("jet"),
#          colorbar=True,)
#plt.legend()
#plt.show()

corr_matrix = df_housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
print(corr_matrix)