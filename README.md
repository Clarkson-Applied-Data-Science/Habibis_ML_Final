# Habibis_ML_Final

## Author: Sheida Habibi
## Title
 House price prediction in the United States of America.

## Content table
|  Number  |    Content  |
|-----|-----|
|1|  [ Description ](#desc)   |
|2|   [ Summary](#meth)   |
|3|    [ Data Gathering and Prepration ](#dg)   |
|3.1|    [ Importing Datasets ](#ld)   |
|3.1.2|    [ Preparing the realstate data set ](#dp)   |
|3.1.3|    [Clustering](#cl)   |
|3.2|    [ Preparing and adding Other datasets ](#anc)   |
|4|   [ Final DataFrame ](#fi)    |
|5|   [ Outliers ](#ou)    |
|6|   [ Correlation Coefficient Matrix ](#cc)    |
|7|   [ Visualization ](#vs)    |
|8|   [ Training, validation and accuracy ](#tr)    |
|9|   [ Accuracy of model for the test data ](#te)    |
|10|   [Conducting model using PCA](#pc)    |
|11|  [ Conclusions ](#con)     |
|12|  [ Limitations ](#li)     |



<a name="desc"></a>
# 1. Description
In this study, we are interested in finding the best model to predict the house price using total land area, population density, house size, number of bedrooms and bathrooms in the different counties in the US. 

<a name="meth"></a>
# 2. Summary
The work involves collecting data on land area, population, real estate, and income from various sources. The collected data is then preprocessed, with the real estate dataset being segmented into three clusters based on average zip code prices. The real estate dataset is merged with population, state area, and income data on the shared column, and important columns are selected and formatted. Data cleaning, outlier handling, feature correlation analysis, feature selection, and data normalization are performed. Machine learning models, including linear regression and random forest, are used on the train/test split data, with accuracy determined through k-fold cross-validation. The same process is repeated using the PCA technique.

<a name="dg"></a>
# 3 Data Gathering and Prepration
The data, about land area, population, real estate, and income, are used and prepared for the furthur analysis.


<a name="ld"></a>
# 3.1.1 Importing Datasets

**Libraries:**
In This project different libraries are being used. All needed packages are loaded:

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

```


**All data sets are read and converted to a data frame format from CSV files using **pandas**:**
```python
real_state = pd.read_csv("realtordata.csv")
income = pd.read_csv("Income.csv")
population = pd.read_csv("population.csv")
land = pd.read_csv("land.csv")
```
**The following data set is used to find the county for the real state data set using zipcodes:**
```python
geo_data= pd.read_csv("geo-data.csv")
```

**The following CSV file is being used to convert complete State names to abbreviation type(state names should be unified):**
```python
st_abr = pd.read_csv("state_abr.csv")
```

<a name="dp"></a>

# 3.1.2 Prepating the realstate data set
At this step,the realstate dataset needs to be prepared and be divided into three clusters based on the average price per ft fot zipcodes.

**This step contains the following content:**
* Addressing missing values and outliers
* choosing important columns
* Changing formats
* Dividing our data to three clusters

## Real state Data Frame

In this data frame price, state, number of bedrooms, number of bathrooms and house size are important. For this purpose, these three columns and zipcode column is chosen.

**The zip code column, as well as state column is later required to find counties from geo_data data frame.**

```python
real_state_data=pd.DataFrame (real_state_data , columns =['price','zip_code', 'bed', 'bath', 'acre_lot','house_size'])
```
Using the following code all **missing values** are being dropped:
```python
real_state_data=real_state_data.dropna(axis=0)
```
At this point data frame containing real state information and the data frame that contains zip codes and county names should be merge.

**For this purpose we should check that whether the type of those two zip codes are the same or not.**

```python
real_state_data.dtypes
```
##### Result:


zip_code      float64


* Using the above code, it can be seen that zipcode in real state data frame is **float64**.

The type of data in geo_data should be checked:
```python
geo_data.dtypes
```
#### Result:


zipcode       object


* As the type of zip code columns is different in the two data frames, we cannot merge them. Therefore, we need to change both columns' type to str for further joining step.
* Since there is no need to do any calculation on the zip code column, it should not have a numeric type, and str would be an appropriate type to choose.

Using the code below all types related to zip code are converted to **string** type:

```python
real_state_data['zip_code'] = real_state_data['zip_code'].astype(int)
real_state_data['zip_code'] = real_state_data['zip_code'].astype(str)
geo_data['zipcode']=geo_data['zipcode'].astype(str)
```


## Converting **zip codes** to **county names**

### Barrier: 
Zip codes should be five-digit numbers. However, at the above data frame, the zeros at begining of the zip codes had been eliminated(many of the zip codes are three and four digits).
### How to fix: 
By using the below code, the data frame is converted to a list, and by using a for loop code, zeros are added to the beginning of the zip codes that have less than five digits:

```python
lst=real_state_data.zip_code.tolist()
i=0
for zip in lst:
    n=5-len(zip)
    for j in range(n):
        zip='0'+zip   
    lst[i]=zip
    if i%1000==0:
        print(i)
    i+=1 
```

The above list, containing the modified version of the zip codes, is added to dataframe. With this modification, all the zipcodes are 5 digit numbers, as can be seen below:

```python
real_state_data['zip_code']=lst
real_state_data
```
![image](https://user-images.githubusercontent.com/113566650/207184761-4da0b4e7-edec-4cfe-b6b4-3c16f11c8f91.png)


## Merging real state and geo_data: 
Berfore merging them the name of zip code column should be change in real state data frame to make it similar tp the one that we have in data fram. By doing this we will be able do merging by using pandas:

```python
real_state_data.rename(columns = {'zip_code':'zipcode'}, inplace = True)
real_state_bycounty=real_state_data.merge(geo_data, how='inner', on='zipcode')
```

#### Doing group by to have information of real satate for each county:
At this step, columns that are needed are being chosen from real state data frame, and the group by function is being applied to have each county name and the average of price and house for size for each of them:




<a name="cl"></a>
# 3.1.3. Clustering

The location of a house is a crucial factor that influences its price. For instance, different cities have a specific price range per square foot, and the cost of a house can vary depending on the location within the city. However, since zipcode (which represent the location) is a categorical variable with many different values , analyzing it poses a challenge. Using one hot encoding on zip codes would be time-consuming and not practical. Therefore, in this project, I grouped the data into three clusters based on zip code. Subsequently, a machine learning algorithm will be trained for each cluster based on other factors to predict the house price.



To cluster zip codes effectively, we need to calculate the average house price per square foot for each zip code. Based on these values, we can then group each zip code into one of three clusters. 
 
So each zipcodes needs to be in:
* Cluster 1 represents the zipcodes with low average price.
* Cluster 2 represents the zipcodes with medium average price.
* Cluster 3 represents the zipcodes with high average price.

**Doing group by to have information of real satate for each zipcodes:**
```python
df1 = pd.DataFrame (real_state_data , columns = ['zip_code', 'price', 'house_size' ])
realstate_grouped = df1.groupby(['zip_code'],as_index=False).mean()
realstate_grouped['price_per_sqft']=realstate_grouped['price']/realstate_grouped['house_size']

 ```
**Clustering the grouped data:**
```python
df1 = pd.DataFrame (real_state_data , columns = ['zip_code', 'price', 'house_size' ])
realstate_grouped = df1.groupby(['zip_code'],as_index=False).mean()
realstate_grouped['price_per_sqft']=realstate_grouped['price']/realstate_grouped['house_size']


X = realstate_grouped[['price_per_sqft']].values

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

n_clusters=3
kmeans = KMeans(n_clusters=n_clusters, random_state=10)

kmeans.fit(X_std)

labels = kmeans.predict(X_std)

for n, i in enumerate(labels):
    if i==0:
        realstate_grouped.loc[n,'Cluster']=i+1
    elif i==1:
        realstate_grouped.loc[n,'Cluster']=i+2
    elif i==2:
        realstate_grouped.loc[n,'Cluster']=i
        
 ```
 
## Clustering visualization

 ```python
 colors = ['red', 'green', 'blue']
plt.scatter(realstate_grouped.index,X_std[:,0], c=labels)
plt.ylabel('Average House Price (standardized)')
plt.show()
```

![clu](https://user-images.githubusercontent.com/113566650/234925983-b26e442f-d48e-417b-8992-48b8c1bc2f5e.png)


 ```python
plt.hist(X[labels==0], label='Cluster 1')
plt.hist(X[labels==2], label='Cluster 2')
plt.hist(X[labels==1],label='Cluster 3')
plt.xlabel('Average House Price')
plt.ylabel('Count')
plt.legend()
plt.show()
 ```
![Cluster](https://user-images.githubusercontent.com/113566650/234928813-9c7064ff-c8e7-4ddb-830e-c527d6ceab3d.png)


## Adding the found clusters to the main real state data frame to see each rows represents which cluster.

 ```python
realstate_grouped=realstate_grouped.loc[:,['zip_code','Cluster']]
realstate_clustered=real_state_data.merge(realstate_grouped, how='inner', on=['zip_code'])
 ```
<a name="anc"></a>
# 3.2. Preparing and adding Other datasets

 ## Income Data Frame
 columns that are important for this study, from income data frame, are being chosen
```python
income=pd.DataFrame (income , columns =['State','County','PerCapitaInc']) 
# Dropping missing values:
income=income.dropna(axis=0)
```
 ### Merging real state and Income data:
```python 
realstate_income=realstate_clustered.merge(income, how='inner', on=['State','County'])
```



## Land Data Frame
Important columns from the land data are picked and columns names that are different from other data sets are chaneged:

```python 
df3 = pd.DataFrame (land , columns = ['USPS','NAME', 'ALAND' ])
df3.rename(columns = {'USPS':'State'}, inplace = True)
df3.rename(columns = {'NAME':'County'}, inplace = True)
```
**'county'** word is added after county names. However, in other data sets, we only have only county names. So using the following code we split by ' county' to remove the **'county'** word for all rows:

```python 
i=0
for c in df3.County:
    Cnt=c.split(' County')[0]
    df3.loc[i,'County']=Cnt
    i+=1
df3
```


 ### Merging real state and Income with land data:

```python 
realstate_income_land=realstate_income.merge(df3, how='inner', on=['State','County'])
``` 


## Population Data Frame

```python 
df4 = pd.DataFrame (population , columns = ['State', 'CountyName', 'TotalPop' ])
``` 
In the population data frame, state names are being written fully, however the abbriviation type of state names are being written in other data frames. so we should change that by joing that with t_abr data frame:
```python 
df4=df4.merge(st_abr, how='inner', on='State')
``` 
Then we remove 'county' word after county name (the same as above):

```python 
i=0
for c in df4.CountyName:
    Cnt=c.split(' County')[0]
    df4.loc[i,'CountyName']=Cnt
    i+=1
``` 

Dropping missing values and changing the column names to make them consistant among all tables:
```python 
population=df4.drop(['State'], axis=1)
population.rename(columns = {'CountyName':'County'}, inplace = True)
population.rename(columns = {'Postal':'State'}, inplace = True)
```
#### Merging Population table created withthe combination of real state, Income and land tables:
```python 
realstate_income_land_population=realstate_income_land.merge(population, how='inner', on=['State', 'County'])
```
* At this point all data frames that are needed are merged.

<a name="dfi"></a>
# 4. Final DataFrame:

```python
Final_data=pd.DataFrame (realstate_income_land_population , columns =['price', 'bed', 'bath', 'acre_lot','house_size', 'Cluster', 'PerCapitaInc', 'ALAND', 'TotalPop'])
Final_data.head()
```
![image](https://user-images.githubusercontent.com/113566650/234932866-8c3efa87-788e-4c0d-a3ad-02bcdee772c5.png)


<a name="ou"></a>
# 5. Outliers
Using the following code, outliers which are more that three Z score above/below the mean, are removed.
```python 
dfnn=realstate_income_land_population.copy()
def remove_outliers_zscore(df_column):
    z_scores = stats.zscore(df_column)
    abs_z_scores = abs(z_scores)
    filtered_entries = (abs_z_scores < 3)
    return df_column[filtered_entries]

realstate_income_land_population[['bed', 'bath','acre_lot','house_size','Price_perft2']] = realstate_income_land_population[['bed', 'bath','acre_lot','house_size','Price_perft2']].apply(remove_outliers_zscore)

realstate_income_land_population=realstate_income_land_population.dropna()
```


<a name="cc"></a>
# 6. Correlation Coefficient Table
```python
correlation = realstate_income_land_population.drop(columns=['Cluster','Price_perft2']).corr()
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(correlation,annot=True)
```

![corr](https://user-images.githubusercontent.com/113566650/234926264-187ec839-8de8-45ff-81e7-b1a94ac20dd5.png)


 **Correlation coefficients measure the strength of the relationship between two variables.**


Abstract of the corrolation metrix can be seen below:

```python
abs(correlation['price']).sort_values(ascending=False)

```

# Result
price           1.000000
house_size      0.622279
bath            0.615662
bed             0.330089
TotalPop        0.278107
PerCapitaInc    0.258289
ALAND           0.125132
acre_lot        0.091766

**Interpretation:**

<a name="vs"></a>
# 7. Visualization:

1
```python
ealstate_income_land_population['size_bin'] = pd.cut(realstate_income_land_population['house_size'], bins=10)
grouped_data = realstate_income_land_population.groupby(['Cluster', 'size_bin'])['price'].mean().reset_index()
pivot_data = grouped_data.pivot(index='size_bin', columns='Cluster', values='price')

ax=pivot_data.plot(kind='bar')
plt.xlabel('House Size Bins')
plt.ylabel('Average Price')
ax.legend(labels=['Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.show()
```
![1](https://user-images.githubusercontent.com/113566650/234927730-a648223c-84a7-4609-9817-65e187a17929.png)

2
```python
grouped = realstate_income_land_population.groupby(['Cluster','bath']).mean().reset_index()
fig, ax = plt.subplots()
colors = ['red', 'green', 'blue']
for i, c in enumerate([1, 2, 3]):
    subset = grouped[grouped['Cluster'] == c]
    ax.scatter(subset['bath'], subset['price'], color=colors[i], label=f'Cluster {c}')
    plt.xlabel("Number of Baths")
    plt.ylabel("Mean Price")
ax.legend()
plt.show()
```
![2](https://user-images.githubusercontent.com/113566650/234927787-89953c57-f7f3-46f3-b66a-845e510d8651.png)

3
```python
grouped = realstate_income_land_population.groupby(['Cluster','bed']).mean().reset_index()
fig, ax = plt.subplots()
colors = ['red', 'green', 'blue']
for i, c in enumerate([1, 2, 3]):
    subset = grouped[grouped['Cluster'] == c]
    ax.scatter(subset['bed'], subset['price'], color=colors[i], label=f'Cluster {c}')
    plt.xlabel("Number of Bedrooms")
    plt.ylabel("Mean Price")
ax.legend()
plt.show()
```
![3](https://user-images.githubusercontent.com/113566650/234927853-479f2ab4-8dba-4956-bfb7-75f00fe7e91a.png)




<a name="tr"></a>
# 8. Training, validation and accuracy:
Machine learning analysis is performed to evaluate the performance of two different regression models (Linear Regression and Random Forest) on the different clusters of a dataset. First, the dataset has been split into training-validation and testing sets(80% train and 20% test). Then the training-validation dataset is split into four folds, and KFold cross-validation is used to evaluate the performance of the regression models on the training data. The KFold function is used to generate the indices for the training and validation sets, and then the data is split into the training and validation sets using these indices. The data are then scaled, and the models are then trained on the scaled training data, and the R2 score is calculated for the validation set for each fold. 

**Here the following steps are taken for each clusters:**

* Splitting dataset

* 4-fold cross validation

* Data Scaling

* fitting random forest and linear regression models

* Finding accuracy



```python
# number of folds
k = 4

kf = KFold(n_splits=k, shuffle=True, random_state=10)

r2_linreg_test_list = []
r2_rf_test_list = []
#r2_svm_test_list = []

for cluster in range(1, n_clusters+1):
    X_cluster = Final_data[Final_data['Cluster'] == cluster].drop(['Cluster','price'], axis=1)
    y_cluster = Final_data[Final_data['Cluster'] == cluster]['price']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=10)
    
    # Evaluation metrics
    r2_linreg_list = []
    r2_rf_list = []
    #r2_svm_list = []
    
    # Loop through each fold
    for train_idx, val_idx in kf.split(X_train_val):
        
        # Split the data into training and validation sets
        X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
        y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
        
        # Scale the features 
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Fit a linear regression model to the data
        linreg_model = LinearRegression()
        linreg_model.fit(X_train_scaled, y_train)
        y_linreg_pred = linreg_model.predict(X_val_scaled)
        
        # Fit a random forest model to the data
        rf_model = RandomForestRegressor(n_estimators=100, random_state=10)
        rf_model.fit(X_train_scaled, y_train)
        y_rf_pred = rf_model.predict(X_val_scaled)
        
        # Fit a SVM model to the data
        #svm_model = SVR(kernel='linear')
        #svm_model.fit(X_train_scaled, y_train)
        #y_svm_pred = svm_model.predict(X_val_scaled)
        
        # Calculate R2 for each model and append to the respective list
        r2_linreg_list.append(r2_score(y_val, y_linreg_pred))
        r2_rf_list.append(r2_score(y_val, y_rf_pred))
        #r2_svm_list.append(r2_score(y_val, y_svm_pred))
    
    #Prediction of the test set
    X_test_scaled = scaler.transform(X_test)
    
    y_linreg_test_pred = linreg_model.predict(X_test_scaled)
    r2_linreg_test_list.append(r2_score(y_test, y_linreg_test_pred))
    
    y_rf_test_pred = rf_model.predict(X_test_scaled)
    r2_rf_test_list.append(r2_score(y_test, y_rf_test_pred))
    
    #y_svm_test_pred = svm_model.predict(X_test_scaled)
    #r2_svm_test_list.append(r2_score(y_test, y_svm_test_pred))
    
    
    # Print the mean R2 for validation set
    print(f"Cluster {cluster}:")
    print(f"   Validation Mean R2 Linear Regression = {sum(r2_linreg_list)/k:.3f}")
    print(f"   Validation Mean R2 Random Forest = {sum(r2_rf_list)/k:.3f}")
    #print(f"  Validation Mean R2 SVM = {sum(r2_svm_list)/k:.3f}")
```
### Result:

**Cluster 1:**

   Validation Mean R2 Linear Regression = 0.546
   
   Validation Mean R2 Random Forest = 0.982
   
**Cluster 2:**

   Validation Mean R2 Linear Regression = 0.671
   
   Validation Mean R2 Random Forest = 0.981
   
**Cluster 3:**

   Validation Mean R2 Linear Regression = 0.890
   
   Validation Mean R2 Random Forest = 0.983
   
   
**Mean R^2 for three clusters: **

 Test Mean R2 Linear Regression = 0.697
 
Test Mean R2 Random Forest = 0.982

   
**Interpretation**
The results show that the random forest model performs consistently well across all three clusters, with validation mean R2 values ranging from 0.981 to 0.983. The linear regression model performs relatively well for clusters 1 and 2, with validation mean R2 values of 0.546 and 0.671, respectively. However, for cluster 3, the linear regression model performs significantly better, with a validation mean R2 of 0.890. For the test set, the random forest model again performs consistently well, with a mean R2 of 0.982. The linear regression model also performs relatively well, with a mean R2 of 0.697.
 
 * Note: As the data was fairly large, Svm technique was too time-consuming to run, it was excluded from the analysis as it and commented out in the code.  
   

<a name="te"></a>
# 9. Accuracy of model for the test data:

```python
# Print the mean R2 for test set
print('\n')
print(f"Test Mean R2 Linear Regression = {sum(r2_linreg_test_list)/n_clusters:.3f}")
print(f"Test Mean R2 Random Forest = {sum(r2_rf_test_list)/n_clusters:.3f}")
#print(f"Test Mean R2 SVM = {sum(r2_svm_test_list)/n_clusters:.3f}")
```
### Result
Test Mean R2 Linear Regression = 0.697
Test Mean R2 Random Forest = 0.982

 Here the accuracy of the test set is calculated and as can be seen it is fairly high number.
 
**Interpretation:** Since accuracy for both train and test data set high and at the same range, we can mae sure that overfitting ha not occured.

<a name="pc"></a>
# 10. Conducting model using PCA:

PCA is added to the code above to select the most relevant features in each cluster. Specifically, the number of variables that account for 90% of the data variation within each cluster is selected. This allows us to reduce the dimensionality of the data while retaining most of the important information.

```python
scaler = StandardScaler()

# number of folds
k = 4

kf = KFold(n_splits=k, shuffle=True, random_state=10)

r2_linreg_test_list = []
r2_rf_test_list = []
#r2_svm_test_list = []


for cluster in range(1, n_clusters+1):
    
    
    X_cluster = Final_data[Final_data['Cluster'] == cluster].drop(['Cluster','price'], axis=1)
    y_cluster = Final_data[Final_data['Cluster'] == cluster]['price']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=10)
    
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    
    #Use PCA to reduce the number of features
    pca = PCA()
    X_train_val_pca = pca.fit_transform(X_train_val_scaled)
    
    variance_ratio = pca.explained_variance_ratio_
    n_components = 0
    explained_variance = 0
    for i in range(len(variance_ratio)):
        explained_variance += variance_ratio[i]
        n_components += 1
        if explained_variance >= 0.9:
            break
    print(f"Cluster {cluster}: {n_components} principal components explain 90% of the variance")
    
    pca = PCA(n_components=n_components)
    X_train_val_pca = pca.fit_transform(X_train_val)
    X_test_pca = pca.transform(X_test)
    
    
    # Evaluation metrics
    r2_linreg_list = []
    r2_rf_list = []
    #r2_svm_list = []
    
    # Loop through each fold
    for train_idx, val_idx in kf.split(X_train_val_pca):
        
        # Split the data into training and validation sets
        X_train, X_val = X_train_val_pca[train_idx], X_train_val_pca[val_idx]
        y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
        
        # Scale the features 
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Fit a linear regression model to the data
        linreg_model = LinearRegression()
        linreg_model.fit(X_train_scaled, y_train)
        y_linreg_pred = linreg_model.predict(X_val_scaled)
        
        # Fit a random forest model to the data
        rf_model = RandomForestRegressor(n_estimators=100, random_state=10)
        rf_model.fit(X_train_scaled, y_train)
        y_rf_pred = rf_model.predict(X_val_scaled)
        
        # Fit a SVM model to the data
        #svm_model = SVR(kernel='linear')
        #svm_model.fit(X_train_scaled, y_train)
        #y_svm_pred = svm_model.predict(X_val_scaled)
        
        # Calculate R2 for each model and append to the respective list
        r2_linreg_list.append(r2_score(y_val, y_linreg_pred))
        r2_rf_list.append(r2_score(y_val, y_rf_pred))
        #r2_svm_list.append(r2_score(y_val, y_svm_pred))
    
    #Prediction of the test set
    X_test_scaled = scaler.transform(X_test_pca)
    
    y_linreg_test_pred = linreg_model.predict(X_test_scaled)
    r2_linreg_test_list.append(r2_score(y_test, y_linreg_test_pred))
    
    y_rf_test_pred = rf_model.predict(X_test_scaled)
    r2_rf_test_list.append(r2_score(y_test, y_rf_test_pred))
    
    #y_svm_test_pred = svm_model.predict(X_test_scaled)
    #r2_svm_test_list.append(r2_score(y_test, y_svm_test_pred))
    
    
    # Print the mean R2 for validation set
    print(f"   Validation Mean R2 Linear Regression = {sum(r2_linreg_list)/k:.3f}")
    print(f"   Validation Mean R2 Random Forest = {sum(r2_rf_list)/k:.3f}")
    #print(f"  Validation Mean R2 SVM = {sum(r2_svm_list)/k:.3f}")
    
    
# Print the mean R2 for test set
print('\n')
print(f"Test Mean R2 Linear Regression = {sum(r2_linreg_test_list)/n_clusters:.3f}")
print(f"Test Mean R2 Random Forest = {sum(r2_rf_test_list)/n_clusters:.3f}")
#print(f"Test Mean R2 SVM = {sum(r2_svm_test_list)/n_clusters:.3f}")

```
### Result:

**Cluster 1: 5 principal components explain 90% of the variance**

   Validation Mean R2 Linear Regression = 0.497
   
   Validation Mean R2 Random Forest = 0.978
   
**Cluster 2: 5 principal components explain 90% of the variance**

   Validation Mean R2 Linear Regression = 0.655
   
   Validation Mean R2 Random Forest = 0.979
   
**Cluster 3: 4 principal components explain 90% of the variance**

   Validation Mean R2 Linear Regression = 0.875
   
   Validation Mean R2 Random Forest = 0.956
   


**Mean R^2 for three clusters:**

Test Mean R2 Linear Regression = 0.671

Test Mean R2 Random Forest = 0.949


**Interpretation:** 
In clusters 1 and 2, five principal components are selected, while in cluster 3, four principal components are used. A slight decrease in R2 for both linear regression and random forest in all clusters is observed compared to the previous results without PCA. This suggests that the PCA approach may not be the most suitable feature selection method in this dataset.

<a name="con"></a>
# 11. Conclusions:
In this project:

1. Clustering results in better predictions: By clustering the data into three distinct groups based on the zip codes of house location, we were able to build separate models for each cluster, which improved the accuracy of our predictions. This suggests that different locations may have different underlying relationships between their features and sale prices.

2. Random forest is better than linear regression: Across all clusters, the random forest model consistently outperformed the linear regression model in terms of R2 score. This indicates that the random forest algorithm is better able to capture complex relationships between features and target variables, and can be a more effective tool for predicting sale prices.

3. PCA does not result in improvement: We applied PCA to reduce the number of features and capture the most relevant information. However, the results showed that this did not lead to a significant improvement in model performance. This could be because the original feature set was already well-suited for prediction, or because the loss of information during the dimensionality reduction process outweighed the benefits.

4. Overall, we can achieve very decent results of price prediction by using random forest: The average R2 score for the random forest model was fairly high, suggesting that the model is effective at predicting sale prices. 

<a name="li"></a>

# 12. Limitation:

While this study provides valuable insights into predicting house prices using machine learning models, it has some limitations. One of the main limitations is the absence of certain crucial features, such as the year of build, distance to city center, and nearby amenities like shopping centers, which could impact the accuracy of the predictive models. Additionally, we only have data from a limited number of cities and states in the US, which may not be representative of the entire population of houses in the US and could limit the generalizability of the results. Another limitation of this study is that the dataset only includes information on houses sold within a specific time frame, which may not represent the current housing market. Finally, while our models achieved decent predictive performance, this study only focuses on using two specific machine learning algorithms (linear regression and random forest) and does not explore other possible algorithms that may produce better results.

