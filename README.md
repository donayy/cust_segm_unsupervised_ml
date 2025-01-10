# Customer Segmentation with Unsupervised Learning

Unsupervised Learning in machine learning refers to a type of learning where the model is provided with data that has no labels or predefined outcomes. 
The algorithm is tasked with finding patterns, structures, or relationships within the data on its own, without guidance.

Key Characteristics:
No labeled data: Unlike supervised learning, where each input is associated with a correct output label, unsupervised learning works with data that has no labels or target outputs.
Pattern discovery: The model tries to identify the underlying structure in the data, such as grouping similar data points (clustering) or reducing the dimensionality of the data (dimensionality reduction).

Common Techniques:
Clustering: Grouping data into clusters based on similarity, e.g., K-Means, DBSCAN.
Dimensionality Reduction: Reducing the number of features while retaining essential information, e.g., PCA (Principal Component Analysis), t-SNE.
Anomaly Detection: Identifying unusual or outlying data points in a dataset.
Association Rule Learning: Finding relationships between variables in large datasets, e.g., Apriori algorithm.

Example Use Cases:
Customer segmentation: Grouping customers based on purchasing behavior without pre-defined categories.
Market basket analysis: Discovering associations between products frequently bought together.
Image compression: Reducing the number of features while preserving image quality.
Unsupervised learning is particularly useful for exploratory data analysis, discovering hidden patterns, and dimensionality reduction tasks.

The company wants to divide its customers into segments and determine marketing strategies according to these segments.

To this end, customer behaviors need to be defined and groups need to be created according to clusters in these behaviors.

12 Variables - 19,945 Observations
master_id : Unique customer number
order_channel : Which channel is used for the shopping platform (Android, ios, Desktop, Mobile)
last_order_channel : Channel where the last shopping was done
first_order_date : Date of the customer's first shopping
last_order_date : Date of the customer's last shopping
last_order_date_online : Date of the customer's last shopping on the online platform
last_order_date_offline : Date of the customer's last shopping on the offline platform
order_num_total_ever_online : Total number of shoppings made by the customer on the online platform
order_num_total_ever_offline : Total number of shoppings made by the customer offline
customer_value_total_ever_offline : Total fee paid by the customer for offline shopping
customer_value_total_ever_online : Total fee paid by the customer for online shopping
interested_in_categories_12 : Last List of categories shopped in the last 12 months
