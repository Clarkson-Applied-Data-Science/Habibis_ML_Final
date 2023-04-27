# Habibis_ML_Final

## Author: Sheida Habibi
## Title
 Price prediction in the United States of America

## Content table
|  Number  |    Content  |
|-----|-----|
|1|  [ Description ](#desc)   |
|2|   [ Summary](#meth)   |
|3|    [ Data Gathering and Prepration ](#dg)   |
|3.1|    [ Importing Datasets ](#ld)   |
|2.1.2|    [ Preparing the realstate data set ](#dp)   |
|2.1.3|    [Clustering](#cl)   |
|2.2|    [ Preparing and adding Other datasets ](#anc)   |
|3|   [ Ultimate data frame](#ud)    |
|4|   [ Correlation Cefficient Table ](#cc)    |
|5|   [ Visualization ](#vs)    |
|5|   [ Machine Learning Model ](#ml)    |
|5|   [ Training, validation and accuracy ](#tr)    |
|7|   [ Accuracy of model for the test data ](#te)    |
|7|   [Conducting model using PCA](#pca)    |
|7|   [Compare the result with and without PCA](#c0)    |
|8|  [ Conclusions ](#con)     |
|9|  [ Limitations ](#le)     |



<a name="desc"></a>
# 1. Description
In this study, we are interested in finding the best model to predict the house price using total land area, population density, house size, number of bedrooms and bathrooms in the different counties in the US. 

<a name="meth"></a>
# 2. Summary
The work involves collecting data on land area, population, real estate, and income from various sources. The collected data is then preprocessed, with the real estate dataset being segmented into three clusters based on average zip code prices. The real estate dataset is merged with population, state area, and income data on the shared column, and important columns are selected and formatted. Data cleaning, outlier handling, feature correlation analysis, feature selection, and data normalization are performed. Machine learning models, including linear regression and random forest, are used on the train/test split data, with accuracy determined through k-fold cross-validation. The same process is repeated using the PCA technique.

<a name="dg"></a>
## 2.1. Data Gathering and Prepration

 
At this step,the realstate dataset needs to be prepared and be divided into three clusters based on the average price per ft fot zipcodes.

<a name="ld"></a>
# 2.1.1 Importing Datasets

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


All data sets are read and converted to a data frame format from CSV files using **pandas**:
```python
real_state = pd.read_csv("realtordata.csv")
income = pd.read_csv("Income.csv")
population = pd.read_csv("population.csv")
land = pd.read_csv("land.csv")
```
The following data set is used to find the county for the real state data set using zipcodes:
```python
geo_data= pd.read_csv("geo-data.csv")
```

The following CSV file is being used to convert complete State names to abbreviation type(state names should be unified):
```python
st_abr = pd.read_csv("state_abr.csv")
```

<a name="dp"></a>

# 2.1.2 Prepating the realstate data set
This step contains the following content:
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


Using the above code, it can be seen that zipcode in real state data frame is **float64**.

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

Cheking the result of real state data frame:
```python
real_state_data
```
#### Result:
![image](https://user-images.githubusercontent.com/113566650/206938786-62dfa2bb-da7a-4e24-8b20-4724940fd543.png)

The resulting real-state data frame can be seen above.

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
```python
df1 = pd.DataFrame (real_state_bycounty , columns = ['county','state_abbr', 'price', 'house_size' ])
realstate_grouped = df1.groupby(['state_abbr','county'],as_index=False).mean()
realstate_grouped.rename(columns = {'state_abbr':'State'}, inplace = True)
```
#### Result:
![image](https://user-images.githubusercontent.com/113566650/206940525-025087a0-8969-4ae9-9c58-09135b58e7e7.png)


<a name="cl"></a>
# 2.1.3. Clustering

since there is a large variation in the price of houses located in differeny areas, we try to combine the locations with similar average price per area, to be able to fit a model tothose three different groups. 
So:
* Cluster 1 represents the zipcodes with low average price.
* Cluster 2 represents the zipcodes with low average price.
* Cluster 3 represents the zipcodes with low average price.

```python
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



 ```python
plt.hist(X[labels==0], label='Cluster 1')
plt.hist(X[labels==2], label='Cluster 2')
plt.hist(X[labels==1],label='Cluster 3')
plt.xlabel('Average House Price')
plt.ylabel('Count')
plt.legend()
plt.show()
 ```


## Adding the found clusters to the main real state data frame to see each rows represents which cluster.

 ```python
realstate_grouped=realstate_grouped.loc[:,['zip_code','Cluster']]
realstate_clustered=real_state_data.merge(realstate_grouped, how='inner', on=['zip_code'])
 ```
<a name="anc"></a>
# 2.2. Preparing and adding Other datasets

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
df3
```
![image](https://user-images.githubusercontent.com/113566650/206941539-3f38903e-bef4-4989-892d-f3f892150e51.png)

As it can be seen, **'county'** word is added after county names. However, in other data sets, we only have only county names. So using the following code we split by ' county' to remove the **'county'** word for all rows:

```python 
i=0
for c in df3.County:
    Cnt=c.split(' County')[0]
    df3.loc[i,'County']=Cnt
    i+=1
df3
```
![image](https://user-images.githubusercontent.com/113566650/207186690-82550846-1cf6-4fc6-b3b3-935578e19bf7.png)


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
![image](https://user-images.githubusercontent.com/113566650/207188617-205e1d0a-9272-4d12-bd01-b79198794010.png)

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




<a name="ot"></a>
# 3. Outliers
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
# 4. Correlation Cefficient Table



 **Correlation coefficients measure the strength of the relationship between two variables.**
```python
correlation = realstate_income_land_population.drop(columns=['Cluster','Price_perft2']).corr()
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(correlation,annot=True)
```



Abstract of the corrolation metrix can be seen below:

```python
abs(correlation['price']).sort_values(ascending=False)
```


<a name="vs"></a>
# 5. Visualization:









<a name="tr"></a>
# 5. Training, validation and accuracy:


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
Cluster 1:
   Validation Mean R2 Linear Regression = 0.546
   Validation Mean R2 Random Forest = 0.982
Cluster 2:
   Validation Mean R2 Linear Regression = 0.671
   Validation Mean R2 Random Forest = 0.981
Cluster 3:
   Validation Mean R2 Linear Regression = 0.890
   Validation Mean R2 Random Forest = 0.983


<a name="te"></a>
# 5. Accuracy of model for the test data:

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



<a name="co"></a>
# 5. Compare the result with and without PCA:




<a name="con"></a>
# 6. Conclusions:
In this study we were able to find correlations between land area, population density, income, house price, and house size in some of counties in the US. We started this task by gathering relevant data in different tables, and performed pre-processing and merging to achieve an ultimate table that contained all of the required data. By finding the correlation coefficient between the different variables and plotting them, we were able to infer the relationships. 

The highest correlation was found between the house price and income per capita, following by the house price and population in that region. This indicates that normally house price is higher in more populated areas with wealthier people. In addition, a very slight correlation between the house size and income per capita was observed, which states that house price is more affected by income than house size. Surprisingly, house size and land area had the least correlation, denoting that houses are not necessarily larger in counties with more available lands.

<a name="li"></a>
# 6. Limitation:


