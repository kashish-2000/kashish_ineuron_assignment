#!/usr/bin/env python
# coding: utf-8

# # Explain how you can implement ML in a real world application.
# 
# Train an SVM regressor on : Bengaluru housing dataset
# 
#                   Must include in details:
# 
#                            - EDA
# 
#                             - Feature engineering 

# In[60]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
bengaluru_data = pd.read_csv('Bengaluru_House_Data.csv')




# In[61]:


bengaluru_data


# In[62]:


# Display basic information about the dataset
print(bengaluru_data.info())

# Display summary statistics
print(bengaluru_data.describe())

# Visualize the distribution of the target variable (e.g., 'price')
plt.figure(figsize=(8, 6))
sns.histplot(bengaluru_data['price'], bins=30, kde=True)
plt.title('Distribution of Housing Prices')
plt.show()

# Explore relationships between features and the target variable
sns.pairplot(bengaluru_data[['total_sqft', 'bath', 'balcony', 'price']])
plt.show()


# In[63]:


# Feature engineering
bengaluru_data['availability_year'] = pd.to_datetime(bengaluru_data['availability'], errors='coerce').dt.year
bengaluru_data.drop(['availability'], axis=1, inplace=True)

# Handle 'total_sqft' column with ranges
def convert_sqft_to_numeric(sqft):
    try:
        # If the value is a range, take the average
        if '-' in sqft:
            tokens = sqft.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        # If the value is a single number, convert it to float
        else:
            return float(sqft)
    except:
        # If there are any issues, return NaN
        return np.nan

# Apply the conversion function to the 'total_sqft' column
bengaluru_data['total_sqft'] = bengaluru_data['total_sqft'].apply(convert_sqft_to_numeric)

# Drop rows with NaN values after the conversion
bengaluru_data = bengaluru_data.dropna()

# Check if the dataset is not empty after preprocessing
if not bengaluru_data.empty:
    # Convert categorical variables to numerical using one-hot encoding
    categorical_cols = ['location', 'size', 'society', 'area_type']
    bengaluru_data = pd.get_dummies(bengaluru_data, columns=categorical_cols, drop_first=True)

    # Feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = bengaluru_data.drop(['price'], axis=1)
    y = bengaluru_data['price']
    X[['total_sqft', 'bath', 'balcony', 'availability_year']] = scaler.fit_transform(X[['total_sqft', 'bath', 'balcony', 'availability_year']])
else:
    print("Dataset is empty after preprocessing. Check your data processing steps.")



# In[54]:





# In[55]:


# Feature engineering
bengaluru_data['availability_year'] = pd.to_datetime(bengaluru_data['availability']).dt.year
bengaluru_data.drop(['availability'], axis=1, inplace=True)


# In[56]:


# Extract features and target variable
X = bengaluru_data.drop(['price'], axis=1)
y = bengaluru_data['price']


# In[57]:


# Convert categorical variables to numerical using one-hot encoding
categorical_cols = ['location', 'size', 'society', 'area_type']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


# In[58]:





# In[ ]:





# In[ ]:




