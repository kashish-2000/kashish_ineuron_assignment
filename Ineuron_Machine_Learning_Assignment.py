#!/usr/bin/env python
# coding: utf-8

# # 1. Imagine you have a dataset where you have different Instagram features like u sername , Caption , Hashtag , Followers , Time_Since_posted , and likes , now your task is to predict the number of likes and Time Since posted and the rest of the features are your input features. Now you have to build a model which can predict the number of likes and Time Since posted. 
# Dataset This is the Dataset You can use this dataset for this question. 

# In[25]:


import pandas as pd


dataset_path = "instagram_reach.csv"
df = pd.read_csv(dataset_path)


# In[26]:


df


# In[27]:


X = df[['USERNAME', 'Caption', 'Hashtags', 'Followers', 'Time since posted']]
y_likes = df['Likes']
y_time_since_posted = df['Time since posted']


# In[28]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_likes_train, y_likes_test, y_time_train, y_time_test = train_test_split(
    X, y_likes, y_time_since_posted, test_size=0.2, random_state=42
)


# In[30]:


from sklearn.preprocessing import LabelEncoder

# Combine X_train and X_test for label encoding
combined_df = pd.concat([X_train, X_test], axis=0)

# Label encode 'Username' for the combined dataset
label_encoder = LabelEncoder()
combined_df['Username_encoded'] = label_encoder.fit_transform(combined_df['USERNAME'])

# Separate back into X_train and X_test
X_train['Username_encoded'] = combined_df['Username_encoded'][:len(X_train)]
X_test['Username_encoded'] = combined_df['Username_encoded'][len(X_train):]

# Drop the original 'Username' column
X_train = X_train.drop('USERNAME', axis=1)
X_test = X_test.drop('USERNAME', axis=1)


# In[32]:


print(y_likes_train.unique())


# In[33]:


# Assuming 'likes' is the target variable
y_likes_train = y_likes_train[pd.to_numeric(y_likes_train, errors='coerce').notnull()]


# In[34]:


y_likes_train = y_likes_train.fillna(y_likes_train.mean())


# In[36]:


import pandas as pd

# Create a DataFrame combining X_train and y_likes_train
train_data = pd.concat([X_train, y_likes_train], axis=1)

# Remove rows with non-numeric values in the 'likes' column
train_data = train_data[pd.to_numeric(train_data['Likes'], errors='coerce').notnull()]

# Separate back into X_train and y_likes_train
X_train = train_data.drop('Likes', axis=1)
y_likes_train = train_data['Likes']


# In[38]:


model_likes.fit(X_train, y_likes_train)


# In[31]:


model_likes.fit(X_train, y_likes_train)
model_time.fit(X_train, y_time_train)


# In[22]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_likes_train, y_likes_test, y_time_train, y_time_test = train_test_split(
    X, y_likes, y_time_since_posted, test_size=0.2, random_state=42
)


# In[23]:


from sklearn.linear_model import LinearRegression

model_likes = LinearRegression()
model_time = LinearRegression()

model_likes.fit(X_train, y_likes_train)
model_time.fit(X_train, y_time_train)


# In[ ]:




