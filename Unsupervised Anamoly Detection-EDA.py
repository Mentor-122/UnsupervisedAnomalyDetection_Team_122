# -*- coding: utf-8 -*-
"""Unsupervised Anamoly Detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AFMmTRunUH6cSoWyDqr07rCvzXMEa615

# PROJECT - **Unsupervised Anamoly Detection**

---
## DATASET - **Healthcare Providers Data For Anomaly Detection**

---
### NAME - **Shrikar Gaikar**

Mount Google Drive
"""

# Mounting Google Drive to access the dataset
from google.colab import drive
drive.mount('/content/drive')

"""Import Libraries"""

# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

"""Load Dataset"""

# Loading the dataset
data = pd.read_csv("/content/drive/MyDrive/Datasets/Healthcare Providers.csv")
data.head()

"""Descriptive Statistics"""

# Displaying descriptive statistics
data.describe()

"""Dataset Information"""

# Displaying information about the dataset
data.info()

"""### Data Preprocessing

---

Convert Object Columns to Numeric Type
"""

# Converting object columns to numeric type where necessary
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

# Verifying data types after conversion
data.info()

"""Handle Missing Values"""

# Checking for missing values and imputing them with the mean
print(data.isnull().sum())

# Imputation of missing values with mean
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Checking for missing values post imputation
print(data.isnull().sum())

"""Check for Duplicate Values"""

# Checking for duplicate values
print(data.duplicated().sum())

"""Merge Name Columns"""

# Merging the name columns into a single column
data['Full Name'] = data['First Name of the Provider'].fillna('') + ' ' + \
                    data['Middle Initial of the Provider'].fillna('') + ' ' + \
                    data['Last Name/Organization Name of the Provider'].fillna('')
data['Full Name'] = data['Full Name'].str.strip()

# Dropping the original name columns
data = data.drop(columns=['Last Name/Organization Name of the Provider',
                          'First Name of the Provider',
                          'Middle Initial of the Provider'])

data.head()

"""Merge Address Columns"""

# Merging the address columns into a single column
data['Full Address'] = data['Street Address 1 of the Provider'].fillna('') + ' ' + \
                       data['Street Address 2 of the Provider'].fillna('')
data['Full Address'] = data['Full Address'].str.strip()

# Dropping the original address columns
data = data.drop(columns=['Street Address 1 of the Provider',
                          'Street Address 2 of the Provider'])

data.head()

"""Standardize Credentials Column"""

# Standardizing the credentials column
data['Credentials of the Provider'] = data['Credentials of the Provider'].str.replace(r'\.', '', regex=True).str.upper()

data.head()

"""## Exploratory Data Analysis (EDA)

---

### 1. Univariate Analysis

---

Distribution of Provider Credentials
"""

# Plotting the distribution of provider credentials
credentials_counts = data['Credentials of the Provider'].value_counts().head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x=credentials_counts.index, y=credentials_counts.values)
plt.title('Distribution of Provider Credentials')
plt.xlabel('Credentials')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

"""Number of Providers by State"""

# Plotting the number of providers by state
state_counts = data['State Code of the Provider'].value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=state_counts.index, y=state_counts.values, palette='rocket')
plt.title('Number of Providers by State')
plt.xlabel('State Code')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

"""Distribution of Provider Types"""

# Plotting the distribution of provider types
provider_type_counts = data['Provider Type'].value_counts().head(20)

plt.figure(figsize=(12, 14))
plt.pie(provider_type_counts, labels=provider_type_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Provider Types')
plt.axis('equal')
plt.show()

# Plotting the top 20 cities of the providers
city_counts = data['City of the Provider'].value_counts().head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x=city_counts.values, y=city_counts.index, palette='viridis')
plt.title('Top 20 Cities of the Providers')
plt.xlabel('Count')
plt.ylabel('City')
plt.show()

"""Distribution of Numeric Columns"""

# Plotting the distribution of numeric columns
numeric_columns = [
    'Number of Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount'
]

plt.figure(figsize=(14, 12))

for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data[column].dropna(), bins=30, kde=True, color='blue')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

"""Correlation Matrix"""

# Plotting the correlation matrix of numerical columns
corr_matrix = data[numeric_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Columns')
plt.show()

"""Pairplot of Numerical Variables"""

# Plotting the pairplot of numerical variables
sns.pairplot(data[numeric_columns])
plt.suptitle('Pairplot of Numerical Variables', y=1.02)
plt.show()

"""### 2. Bivariate Analysis

---

Provider Gender Distribution by State
"""

# Countplot of provider gender distribution by state
plt.figure(figsize=(14, 8))
sns.countplot(x='State Code of the Provider', hue='Gender of the Provider', data=data)
plt.title('Provider Gender Distribution by State')
plt.xlabel('State Code')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Gender')
plt.show()

"""Average Submitted Charge vs. Average Payment"""

# Scatter plot of average submitted charge vs. average payment
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Average Submitted Charge Amount', y='Average Medicare Payment Amount', data=data, hue='Provider Type', alpha=0.8)
plt.title('Average Submitted Charge Amount vs. Average Medicare Payment Amount')
plt.xlabel('Average Submitted Charge Amount')
plt.ylabel('Average Medicare Payment Amount')
plt.legend(title='Provider Type', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

"""Average Medicare Payment Amount by Provider Type"""

# Boxplot of average Medicare payment amount by provider type
plt.figure(figsize=(14, 10))
sns.boxplot(x='Provider Type', y='Average Medicare Payment Amount', data=data)
plt.title('Distribution of Average Medicare Payment Amount by Provider Type')
plt.xlabel('Provider Type')
plt.ylabel('Average Medicare Payment Amount')
plt.xticks(rotation=90)
plt.show()

"""Average Number of Services by State and Gender"""

# Point plot of average number of services by state and gender
plt.figure(figsize=(14, 8))
sns.pointplot(x='State Code of the Provider', y='Number of Services', hue='Gender of the Provider', data=data, dodge=True, markers=["o", "x"], linestyles=["-", "--"])
plt.title('Average Number of Services by State and Gender')
plt.xlabel('State Code')
plt.ylabel('Number of Services')
plt.xticks(rotation=90)
plt.show()

"""Correlation between Number of Services and Payment Amounts"""

# Scatter plot of number of services vs. average Medicare payment amount
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Number of Services', y='Average Medicare Payment Amount', data=data)
plt.title('Number of Services vs. Average Medicare Payment Amount')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.show()

"""Average Submitted Charge Amount by City"""

# Bar graph of average submitted charge amount by city
top_20_cities = data['City of the Provider'].value_counts().head(20).index
filtered_city_data = data[data['City of the Provider'].isin(top_20_cities)]

plt.figure(figsize=(14, 8))
sns.barplot(x='Average Submitted Charge Amount', y='City of the Provider', data=filtered_city_data, estimator=sum)
plt.title('Average Submitted Charge Amount by Top 20 Cities')
plt.xlabel('Average Submitted Charge Amount')
plt.ylabel('City')
plt.show()

"""Distribution of Top 20 Provider Credentials by Gender"""

# Bar graph of the distribution of top 20 provider credentials by gender
data['Credentials of the Provider'] = data['Credentials of the Provider'].fillna('Unknown')
top_20_credentials = data['Credentials of the Provider'].value_counts().head(20).index
filtered_credential_data = data[data['Credentials of the Provider'].isin(top_20_credentials)]

plt.figure(figsize=(14, 10))
sns.countplot(data=filtered_credential_data, x='Credentials of the Provider', hue='Gender of the Provider', order=top_20_credentials)
plt.title('Distribution of Top 20 Provider Credentials by Gender')
plt.xlabel('Credentials')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Gender')
plt.show()

"""Average Medicare Payment Amount by Credentials"""

# Boxplot of average Medicare payment amount by credentials
plt.figure(figsize=(14, 8))
sns.boxplot(x='Credentials of the Provider', y='Average Medicare Payment Amount', data=filtered_credential_data, order=top_20_credentials)
plt.title('Average Medicare Payment Amount by Credentials')
plt.xlabel('Credentials')
plt.ylabel('Average Medicare Payment Amount')
plt.xticks(rotation=90)
plt.show()

"""### 3. Additional Insights

---

Insights on Provider Gender Distribution
"""

# Insights on provider gender distribution
gender_counts = data['Gender of the Provider'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=gender_counts.index, y=gender_counts.values)
plt.title('Provider Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

"""Insights on Provider Type Distribution"""

# Insights on provider type distribution
plt.figure(figsize=(12, 8))
sns.countplot(y='Provider Type', data=data, order=data['Provider Type'].value_counts().index)
plt.title('Provider Type Distribution')
plt.xlabel('Count')
plt.ylabel('Provider Type')
plt.show()

"""Insights on State-wise Distribution of Providers"""

# Insights on state-wise distribution of providers
plt.figure(figsize=(12, 8))
sns.boxplot(x='State Code of the Provider', y='Number of Services', data=data)
plt.title('State-wise Distribution of Providers by Number of Services')
plt.xlabel('State Code')
plt.ylabel('Number of Services')
plt.xticks(rotation=90)
plt.show()

"""Insights on Distribution of Number of Services"""

# Insights on distribution of number of services
plt.figure(figsize=(10, 6))
sns.histplot(data['Number of Services'].dropna(), bins=30, kde=True, color='blue')
plt.title('Distribution of Number of Services')
plt.xlabel('Number of Services')
plt.ylabel('Frequency')
plt.show()