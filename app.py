import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("datasets/data.csv")


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["last_order_date"].max()  # 2021-05-30
analysis_date = dt.datetime(2021, 6, 1)

df["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')
df["tenure"] = (df["last_order_date"]-df["first_order_date"]).astype('timedelta64[D]')

model_df = df[["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
               "customer_value_total_ever_online", "recency", "tenure"]]
model_df.head()

# Customer Segmentation with K-Means 
sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df = pd.DataFrame(model_scaling, columns=model_df.columns)
model_df.head()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show(block=True)


k_means = KMeans(n_clusters=4, random_state=17).fit(model_df)
segments = k_means.labels_
segments


final_df = df[["master_id", "order_num_total_ever_online", "order_num_total_ever_offline",
               "customer_value_total_ever_offline", "customer_value_total_ever_online",
               "recency", "tenure"]]
final_df["segment"] = segments
final_df.head()


final_df.groupby("segment").agg({"order_num_total_ever_online": ["mean", "min", "max"],
                                 "order_num_total_ever_offline": ["mean", "min", "max"],
                                 "customer_value_total_ever_offline": ["mean", "min", "max"],
                                 "customer_value_total_ever_online": ["mean", "min", "max"],
                                 "recency": ["mean", "min", "max"],
                                 "tenure": ["mean", "min", "max", "count"]})

# Customer Segmentation with  Hierarchical Clustering

hc_average = linkage(model_df, "average")

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show(block=True)


plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show(block=True)


segment = AgglomerativeClustering(n_clusters=5, linkage="average")
segments = segment.fit_predict(model_df)

final_df = df[["master_id", "order_num_total_ever_online", "order_num_total_ever_offline",
               "customer_value_total_ever_offline", "customer_value_total_ever_online", "recency", "tenure"]]
final_df["segment"] = segments
final_df["segment"] = final_df["segment"] + 1

final_df.head()


final_df.groupby("segment").agg({"order_num_total_ever_online": ["mean", "min", "max"],
                                 "order_num_total_ever_offline": ["mean", "min", "max"],
                                 "customer_value_total_ever_offline": ["mean", "min", "max"],
                                 "customer_value_total_ever_online": ["mean", "min", "max"],
                                 "recency": ["mean", "min", "max"],
                                 "tenure": ["mean", "min", "max", "count"]})

