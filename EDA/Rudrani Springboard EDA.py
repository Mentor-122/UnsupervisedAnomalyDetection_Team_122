#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">EXPLORATORY DATA ANALYSIS</h1>
# <H3 align="right">Name: Rudrani Ghosh</H3>

# In[375]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("/Users/rudranighosh/Downloads/Healthcare Providers.csv")
data.head()


# In[376]:


# Descriptive statistics
data.describe()


# In[377]:


# information about the dataset
data.info()


# <h2>Converting Object to Numeric Type </h2>

# In[378]:


numeric_columns = [
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')
    
    
data.info()


# <h2>Looking for Missing Values and imputing them with Mean </h2>

# In[379]:


# missing values
print(data.isnull().sum())


# In[380]:


# Imputation of missing values with mean
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

print(data.isnull().sum())


# <h2>Looking for Duplicate Values </h2>

# In[381]:


# Check for duplicates
print(data.duplicated().sum())


# <h2>Data Preprocessing </h2>

# In[382]:


# Merging the name columns into a single column 
data['Full Name'] = data['First Name of the Provider'].fillna('') + ' ' + \
                    data['Middle Initial of the Provider'].fillna('') + ' ' + \
                    data['Last Name/Organization Name of the Provider'].fillna('')
data['Full Name'] = data['Full Name'].str.strip()

data = data.drop(columns=['Last Name/Organization Name of the Provider', 
                          'First Name of the Provider', 
                          'Middle Initial of the Provider'])

data.head()


# In[383]:


# Merging the address columns 
data['Full Address'] = data['Street Address 1 of the Provider'].fillna('') + ' ' + \
                       data['Street Address 2 of the Provider'].fillna('')
data['Full Address'] = data['Full Address'].str.strip()

data = data.drop(columns=['Street Address 1 of the Provider', 
                          'Street Address 2 of the Provider'])

data.head()


# In[384]:


# Standardize credentials
data['Credentials of the Provider'] = data['Credentials of the Provider'].str.replace(r'\.', '', regex=True).str.upper()

data.head()


# <h2>GRAPHS: </h2>
# 

# In[385]:


# Plot bar plot for Credentials of the Provider
credentials_counts = data['Credentials of the Provider'].value_counts().head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x=credentials_counts.index, y=credentials_counts.values)
plt.title('Distribution of Provider Credentials')
plt.xlabel('Credentials')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


# In[386]:


state_counts = data['State Code of the Provider'].value_counts()

# bar graph for State Code of the Provider
plt.figure(figsize=(12, 6))
sns.barplot(x=state_counts.index, y=state_counts.values, palette='rocket')
plt.title('Number of Providers by State')
plt.xlabel('State Code')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


# In[387]:


provider_type_counts = data['Provider Type'].value_counts().head(20)

# pie chart for Provider Types
plt.figure(figsize=(12, 14))
plt.pie(provider_type_counts, labels=provider_type_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Provider Types')
plt.axis('equal')  
plt.show()


# In[388]:


# occurrences of each city
city_counts = data['City of the Provider'].value_counts().head(20)

# Plot of top 20 cities
sns.barplot(x=city_counts.values, y=city_counts.index, palette='viridis')
plt.title('Top 20 Cities of the Providers')
plt.xlabel('Count')
plt.ylabel('City')
plt.show()


# In[389]:


numeric_columns = [
    'Number of Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount'
]

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

plt.figure(figsize=(14, 12))

for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data[column].dropna(), bins=30, kde=True, color='blue')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[390]:


corr_matrix = data[numeric_columns].corr()

#correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Columns')
plt.show()


# In[391]:


sns.pairplot(data[numeric_columns])
plt.title('Pairplot of Numerical Variables')
plt.show()


# <h2>Bivariate Analysis </h2>

# In[392]:


#Countplot of Provider Gender Distribution by State

plt.figure(figsize=(14, 8))
sns.countplot(x='State Code of the Provider', hue='Gender of the Provider', data=data)
plt.title('Provider Gender Distribution by State')
plt.xlabel('State Code')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Gender')
plt.show()


# In[393]:


#Scatter Plot of Average Submitted Charge vs. Average Payment

plt.figure(figsize=(12, 8))
sns.scatterplot(x='Average Submitted Charge Amount', y='Average Medicare Payment Amount', data=data, hue='Provider Type', alpha=0.8)
plt.title('Average Submitted Charge Amount vs. Average Medicare Payment Amount')
plt.xlabel('Average Submitted Charge Amount')
plt.ylabel('Average Medicare Payment Amount')
plt.legend(title='Provider Type', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# In[394]:


#Boxplot of Average Medicare Payment Amount by Provider Type

plt.figure(figsize=(14, 10))
sns.boxplot(x='Provider Type', y='Average Medicare Payment Amount', data=data)
plt.title('Distribution of Average Medicare Payment Amount by Provider Type')
plt.xlabel('Provider Type')
plt.ylabel('Average Medicare Payment Amount')
plt.xticks(rotation=90)
plt.show()


# In[395]:


# point plot to show the relationship between average Number of Services by State Code of the Provider and Gender

plt.figure(figsize=(14, 8))
sns.pointplot(x='State Code of the Provider', y='Number of Services', hue='Gender of the Provider', data=data, dodge=True, markers=["o", "x"], linestyles=["-", "--"])
plt.title('Average Number of Services by State and Gender')
plt.xlabel('State Code')
plt.ylabel('Number of Services')
plt.xticks(rotation=90)
plt.show()



# In[396]:


#Correlation between Number of Services and Payment Amounts

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Number of Services', y='Average Medicare Payment Amount', data=data)
plt.title('Number of Services vs. Average Medicare Payment Amount')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.show()


# In[397]:


#Bargraph of Average Submitted Charge Amount by City:

top_20_cities = data['City of the Provider'].value_counts().head(20).index
filtered_city_data = data[data['City of the Provider'].isin(top_20_cities)]

plt.figure(figsize=(14, 8))
sns.barplot(x='Average Submitted Charge Amount', y='City of the Provider', data=filtered_city_data, estimator=sum)
plt.title('Average Submitted Charge Amount by Top 20 Cities')
plt.xlabel('Average Submitted Charge Amount')
plt.ylabel('City')
plt.show()


# In[398]:


#Bargraph of Distribution of Top 20 Provider Credentials by Gender
data['Credentials of the Provider'] = data['Credentials of the Provider'].str.strip()
data['Gender of the Provider'] = data['Gender of the Provider'].str.strip()
top_20_credentials = data['Credentials of the Provider'].value_counts().head(20).index

# Filtered data to include only rows with the top 20 credentials
filtered_data = data[data['Credentials of the Provider'].isin(top_20_credentials)]

plt.figure(figsize=(14, 8))
sns.countplot(data=filtered_data, x='Credentials of the Provider', hue='Gender of the Provider', order=top_30_credentials)
plt.title('Distribution of Top 20 Provider Credentials by Gender')
plt.xlabel('Credentials')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Gender')
plt.show()


# In[399]:


#Average Medicare Payment Amount by Credentials

plt.figure(figsize=(14, 8))
sns.boxplot(x='Credentials of the Provider', y='Average Medicare Payment Amount', data=filtered_data, order=top_30_credentials)
plt.title('Average Medicare Payment Amount by Credentials')
plt.xlabel('Credentials')
plt.ylabel('Average Medicare Payment Amount')
plt.xticks(rotation=90)
plt.show()


# In[ ]:




