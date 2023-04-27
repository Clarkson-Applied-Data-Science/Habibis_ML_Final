# Habibis_ML_Final

## Author: Sheida Habibi
## Title
 Price prediction in the United States of America

## Content table
|  Number  |    Content  |
|-----|-----|
|1|  [ Description ](#desc)   |
|2|   [ Method ](#meth)   |
|2.1|    [ Data Gathering ](#dg)   |
|2.1.1|    [ Real State Dataset ](#ld)   |
|2.1.2|    [ Data preprocessing ](#dp)   |
|2.1.3|    [ Adding new columns ](#anc)   |
|3|   [ Ultimate data frame](#ud)    |
|4|   [ Correlation Cefficient Table ](#cc)    |
|5|   [ Clustering ](#cc)    |
|6|   [ Train/Test Split ](#cc)    |
|7|   [ Accuracy of model for the train data ](#cc)    |
|8|   [ Accuracy of model for the test data ](#cc)    |
|9|  [ Interpretation and plots ](#inter)     |
|10|  [ Conclusions ](#con)     |
|7|  [ Limitations ](#con)     |



<a name="desc"></a>
# 1. Description
In this study, we are interested in finding the best model to predict the house price using total land area, population density, house size, number of bedrooms and bathrooms in the different counties in the US. 

<a name="meth"></a>
# 2. Method

<a name="dg"></a>
## 2.1. Data Gathering

Data on land area, population, real estate, and income were collected from different sources.
The following steps are needed :
* Preprocessing realstate dataset. 
* Putting real states dataset into three Clustering based on the average price of the zipcodes.
* Merging the realstate dataset, with out datasets about population, states' area and income, on the column that they have in common.
* Selecting important columns and changing some formats.
* Data preprocessing such as Cleaning data sets by removing/replacing the null values and adresing the outliers.
* Finding corrolation between features
* Feature selection
* Normalazing the ddata
* Spliting the data to Train/test and conducting machine learning models(both linear regression and random forest)
* Finding accuaracy for train data and k-fold cross validation.
* Finding accuaracy for test data
* Repeating the same process using PCA technique.

**Note: We should consider that country names might be similar for different states. Therefore, the column on which  we merge should be the combination of county and state. However, for clusteing zipcode is used. **

At this step, our data sets needs to be merged, and essential columns for the study should be chosen.
For this purpose, some formats need to be changed, and missing values should be addressed.

At this step,the realstate dataset needs to be prepared and be divided into three clusters.

<a name="ld"></a>
# 2.2 RealState Dataset

**Libraries:**
In This project different libraries are being used. All needed packages are loaded:

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
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

# 2.3 Preprocessing the realstate data set
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



## Clustering
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




 ## Income Data Frame
 columns that are important for this study, from income data frame, are being chosen
```python
income=pd.DataFrame (income , columns =['State','County','PerCapitaInc']) 
# Dropping missing values:
income=income.dropna(axis=0)
```
 ### Merging real state and Income data:
```python 
realstate_income=realstate_grouped.merge(income, how='inner', on=['State','County'])
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

<a name="anc"></a>
## 2.3 Adding new columns
Addingng new columns using the information that we have to obtain new information. The following columns are obtained:

* Land_perCapita: Area of land/Total population of an area
* Price_perft2: Average price of house/ average house size

```python 
realstate_income_land_population['Land_perCapita']=realstate_income_land_population['ALAND']/realstate_income_land_population['TotalPop']
realstate_income_land_population['Price_perft2']=realstate_income_land_population['price']/realstate_income_land_population['house_size']
```
<a name="ud"></a>
# 3. Ultimate data frame
Here is the ultimate table that is used to investigate on correlation coeficient and to answer the questions that made us to do this study:
```python 
realstate_income_land_population
```
![image](https://user-images.githubusercontent.com/113566650/206943479-d51f51e3-0f80-483c-8c4f-2bc470b63cda.png)


<a name="cc"></a>
# 4. Correlation Cefficient Table
Using the following code, the correlation coefficient between each two variables can be seen:
```python
my_correlation = realstate_income_land_population.corr()
sns.heatmap(my_correlation,annot=True)
```
![output](https://user-images.githubusercontent.com/113566650/206943317-9fd76270-d7f7-4531-bad6-e3162c97a8e3.png)

<a name="inter"></a>
# 5. Interpretation and plots
 **Correlation coefficients measure the strength of the relationship between two variables.**
## Correlation between House price and Income:

The average home price has a moderate correlation with counties' average income per capita. (correlation coefficient: 0.5). It was the highest correlation coefficient in the table.

Here is the code that for scatter plot that represent values for average house price and income per capita:
```python
x1 = realstate_income_land_population['PerCapitaInc'].values
y1 = realstate_income_land_population['price'].values
plt.xlabel("Income per person")
plt.ylabel("Average house price")
plt.scatter(x1, y1)
plt.show()
```
**Result**

![output1](https://user-images.githubusercontent.com/113566650/207165502-fe7680e2-731e-4b03-a3ed-b9f98d936e08.png)

## Correlation between House size and Land area:
There is a very weak correlation between land size and house size for an area(with a correlation coefficient of -0.15). This correlation coefficient is negative, meaning that house size decreases to some extent by increasing land size.
```python
x2 = realstate_income_land_population['ALAND'].values
y2 = realstate_income_land_population['house_size'].values
plt.xlabel("State land area")
plt.ylabel("House size")
plt.scatter(x2, y2)
plt.show()
```
**Result**

![output2](https://user-images.githubusercontent.com/113566650/207168689-28ff6e8c-fc52-4187-92f1-af8dc7dc6104.png)


## Correlation between House size and Income per capita:
The correlation coefficient between house size and income is a weak to moderate(0.27).The following scatter plot represents the over all trend for the value of house size and Income per capita based on the data that we have.

```python
x3 = realstate_income_land_population['PerCapitaInc'].values
y3 = realstate_income_land_population['house_size'].values
plt.xlabel("Income per person")
plt.ylabel("House size")
plt.scatter(x3, y3)
plt.show()
```
**Result**

![output3](https://user-images.githubusercontent.com/113566650/207168928-2b82b54f-e5a8-4a69-8cf9-4dd344ecb4fd.png)


## Correlation between Average house price and Population

The price of a house per square feat has a moderate correlation with the population of that area(correlation coefficient: 0.4)

```python
x6 = realstate_income_land_population['TotalPop'].values
y6 = realstate_income_land_population['Price_perft2'].values
plt.xlabel("Total Population")
plt.ylabel("Price per $ft^2$")
plt.scatter(x6, y6)
plt.show()
```

**Result**

![image](https://user-images.githubusercontent.com/113566650/207193949-ffddfdfe-548f-43e8-a101-105019bacd12.png)

## Correlation between Land area and Population

The correlation coefficient between total population and the land of counties ia a negative weak to moderate relationship.(corelation coeficient= -.024)
```python
x4 = realstate_income_land_population['ALAND'].values
y4 = realstate_income_land_population['TotalPop'].values
plt.xlabel("State Land Area")
plt.ylabel("Total population")
plt.scatter(x4, y4)
plt.show()
```
**Result**

![output7](https://user-images.githubusercontent.com/113566650/207171433-7bb6a5dc-1ae0-43a7-aa13-e0a1b911dac8.png)

<a name="con"></a>
# 6. Conclusions:
In this study we were able to find correlations between land area, population density, income, house price, and house size in some of counties in the US. We started this task by gathering relevant data in different tables, and performed pre-processing and merging to achieve an ultimate table that contained all of the required data. By finding the correlation coefficient between the different variables and plotting them, we were able to infer the relationships. 

The highest correlation was found between the house price and income per capita, following by the house price and population in that region. This indicates that normally house price is higher in more populated areas with wealthier people. In addition, a very slight correlation between the house size and income per capita was observed, which states that house price is more affected by income than house size. Surprisingly, house size and land area had the least correlation, denoting that houses are not necessarily larger in counties with more available lands.
