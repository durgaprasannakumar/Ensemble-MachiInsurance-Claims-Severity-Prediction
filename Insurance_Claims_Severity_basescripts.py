#!/usr/bin/env python
# coding: utf-8

# # Insurance Claim Prediction

# In[1]:


pip install pandas


# In[2]:


pip install --user memory_profiler


# In[3]:


pip install matplotlib


# In[4]:


pip install seaborn


# In[5]:


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
import math
import pickle


# In[6]:


from memory_profiler import profile


# In[7]:


pd.options.display.max_columns = None


# In[8]:


train_data = pd.read_csv(r'C:\Users\prasa\Downloads\dataset\train.csv')


# In[9]:


train_data.shape


# In[10]:


train_data.head()


# In[11]:


column_names = np.array(train_data.columns)
print(column_names)


# ###### Identify the categorical and numerical columns to check the data distribution and 5 point summary

# In[12]:


column_datatypes = train_data.dtypes
categorical_columns = list(column_datatypes[column_datatypes=="object"].index.values)
continuous_columns = list(column_datatypes[column_datatypes=="float64"].index.values)
continuous_columns.remove('loss')


#  ### function to check the distribution of values in categorical columns
# ### Training data and Categorical columns list

# In[13]:


@profile
def category_distribution(train_data,categorical_columns):
    categorical_column_distribution = list()
    for cat_column in categorical_columns:
        categorical_column_distribution.append(train_data[cat_column].value_counts())
    return(categorical_column_distribution)


# In[14]:


categorical_column_distribution = category_distribution(train_data,categorical_columns)


# In[15]:


categorical_column_distribution


# In[16]:


length_categorical_columns = list(map(lambda x:len(x),categorical_column_distribution))


# #### count the number of columns having the same number of unique values

# In[17]:


distribution_dict = dict()


# In[18]:


for val in length_categorical_columns:
    if val in distribution_dict.keys():
        count = distribution_dict[val]
        distribution_dict[val] = count+1
    else:
        distribution_dict[val]=1


# In[19]:


distribution_dict


# #### plot showing the count of columns having same number of unique values

# In[20]:


keys = distribution_dict.keys()
values = distribution_dict.values()
plt.bar(keys, values,width=0.8)
plt.xlabel('Distinct Values in Categorical Variable', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Categorical Labels with Same Unique Values',fontsize=20)
plt.rcParams['figure.figsize'] = [48/2.54, 10/2.54]
plt.show()


# In[21]:


train_data[continuous_columns].describe()


# ### Data cleaning and pre-processing

# #### Check if there is any missing value in the columuns
# #### value of 0 indicates no missing values
#   

# In[22]:


missing_values = train_data.isnull().sum()
np.max(missing_values)


# In[23]:


total_rows = train_data.shape[0]
columns_with_blanks_cat = np.random.randint(1,116,2)
columns_with_blanks_cont = np.random.randint(117,130,3)
columns_with_blank = np.append(columns_with_blanks_cat,columns_with_blanks_cont)


# In[24]:


for col in columns_with_blank:
    rows_with_blanks = np.random.randint(1,total_rows,5)
    train_data.iloc[rows_with_blanks,col] = np.nan


# In[25]:


missing_values = train_data.isnull().sum()
np.max(missing_values)


# In[26]:


columns_with_missing = train_data.columns[train_data.isnull().any()]
print(columns_with_missing)


# In[27]:


@profile
class Data_preprocessing:
    def __init__(self,train_data):
        self.train_data = train_data
        
    def __init__(self,train_data):
        self.train_data = train_data

    def missing_value_continuous(self,column_names_with_specific_type,imputation_type="mean"):
        if imputation_type=="mean":
            mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            mean_imputer.fit(self.train_data[column_names_with_specific_type])
            self.train_data[column_names_with_specific_type]=mean_imputer.transform(self.train_data[column_names_with_specific_type])
        if imputation_type=="median":
            median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
            median_imputer.fit(self.train_data[column_names_with_specific_type])
            self.train_data[column_names_with_specific_type]=median_imputer.transform(self.train_data[column_names_with_specific_type])
        return self.train_data

    def missing_value_categorical(self,column_names_with_specific_type,imputation_type="most_frequent"):
        most_frequent = SimpleImputer(strategy="most_frequent")
        most_frequent.fit(self.train_data[column_names_with_specific_type])
        self.train_data[column_names_with_specific_type] = most_frequent.transform(train_data[column_names_with_specific_type])
        return self.train_data

    def outlier_treatment(self,Q1,Q3,IQR,columns_with_outlier,action):
        if action=="median":
            for i in range(len(columns_with_outlier)):
                column_name = columns_with_outlier[i]
                meadian_outlier = np.median(self.train_data[column_name])
                self.train_data.loc[self.train_data[((self.train_data[column_name]<(Q1[column_name]-(1.5*IQR[column_name])))|(self.train_data[column_name]>(Q3[column_name]+(1.5*IQR[column_name]))))].index,column_name]=meadian_outlier
        if action=="mean":
            for i in range(len(columns_with_outlier)):
                column_name = columns_with_outlier[i]
                mean_outlier = np.mean(self.train_data[column_name])
                self.train_data.loc[self.train_data[((self.train_data[column_name]<(Q1[column_name]-(1.5*IQR[column_name])))|(self.train_data[column_name]>(Q3[column_name]+(1.5*IQR[column_name]))))].index,column_name]=mean_outlier
        if action=="remove":
            for i in range(len(columns_with_outlier)):
                column_name = columns_with_outlier[i]
                self.train_data = self.train_data[~((self.train_data[column_name]<(Q1[column_name]-(1.5*IQR[column_name])))|(self.train_data[column_name]>(Q3[column_name]+(1.5*IQR[column_name]))))]
        return self.train_data


# In[28]:


Data_preprocessing_obj = Data_preprocessing(train_data)
train_data = Data_preprocessing_obj.missing_value_continuous(continuous_columns,"median")
train_data = Data_preprocessing_obj.missing_value_categorical(categorical_columns)


# ##### Section on handling outliers in the dataset

# In[29]:


ax = sns.boxplot(data=train_data[continuous_columns], orient="h", palette="Set2")


# In[30]:


columns_with_outlier = ['cont7','cont9','cont10']


# ### compute the interquartile range for all continuous columns

# In[31]:


Q1 = train_data[continuous_columns].quantile(0.25)
Q3 = train_data[continuous_columns].quantile(0.75)
IQR = (Q3-Q1)
train_data = Data_preprocessing_obj.outlier_treatment(Q1,Q3,IQR,columns_with_outlier,"median")


# In[32]:


ax = sns.boxplot(data=train_data[continuous_columns], orient="h", palette="Set2")


# ### Feature elimination techniques for continuous and categorical features

# In[33]:


@profile
def feature_selection_numerical_variables(train_data,qthreshold,corr_threshold,exclude_numerical_cols_list):
    num_colums = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_columns = list(train_data.select_dtypes(include=num_colums).columns)
    numerical_columns = [column for column in numerical_columns if column not in exclude_numerical_cols_list]
        #remove variables with constant variance

    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(train_data[numerical_columns])
    constant_columns = [column for column in train_data[numerical_columns].columns 
                    if column not in train_data[numerical_columns].columns[constant_filter.get_support()]]
    if len(constant_columns)>0:
        train_data.drop(labels=constant_columns, axis=1, inplace=True)
    #remove deleted columns from dataframe
    numerical_columns = [column for column in numerical_columns if column not in constant_columns]

    #remove variables with qconstant variance
    #Remove quasi-constant variables

    qconstant_filter = VarianceThreshold(threshold=qthreshold)
    qconstant_filter.fit(train_data[numerical_columns])
    qconstant_columns = [column for column in train_data[numerical_columns].columns
                    if column not in train_data[numerical_columns].columns[constant_filter.get_support()]]
    if len(qconstant_columns)>0:
        train_data.drop(labels=qconstant_columns, axis=1, inplace=True)

      #remove deleted columns from dataframe
    numerical_columns = [column for column in numerical_columns if column not in qconstant_columns]
      #remove correlated variables
    correlated_features = set()
    correlation_matrix = train_data[numerical_columns].corr()
    ax = sns.heatmap(
    correlation_matrix,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right');
      #print(correlation_matrix)  
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > corr_threshold:
                colname = correlation_matrix.columns[i]
                colcompared = correlation_matrix.columns[j]
    #check if the column compared against is not in the columns excluded list
                if colcompared not in correlated_features:
                    correlated_features.add(colname)
    train_data.drop(labels=correlated_features, axis=1, inplace=True)

    return train_data,constant_columns,qconstant_columns,correlated_features


# In[34]:


train_data,constant_columns,qconstant_columns,correlated_features =feature_selection_numerical_variables(train_data,0.01,0.75,['loss','id'],)


# In[35]:


correlated_features


# #### Handling correlation between categorical variables

# In[36]:


for cf1 in categorical_columns:
    le = LabelEncoder()
    le.fit(train_data[cf1].unique())
    filename = cf1+".sav"
    pickle.dump(le, open(filename, 'wb'))
    train_data[cf1] = le.transform(train_data[cf1])


# In[37]:


#snippet to calculate the unique values with a categorical columns
df = pd.DataFrame(columns=["Column_Name","Count"])
for cat in categorical_columns:
    unique_value_count = len(train_data[cat].unique())
    df = df.append({'Column_Name': cat, "Count":int(unique_value_count)}, ignore_index=True)
columns_unique_value = np.array(df.Count.value_counts().index)


# In[38]:


columns_unique_value


# In[39]:


df


# In[40]:


#snippet to identify the dependent/correlated categorical variables and drop them
columns_to_drop_cat = set()
correlated_columns = dict()
for unique_value_count in columns_unique_value:
    if unique_value_count>1:
        categorical_columns = df.loc[df.Count==unique_value_count,'Column_Name']
        categorical_columns = categorical_columns.reset_index(drop=True)
        columns_length=len(categorical_columns)
        for col in range(columns_length-1):
            column_to_compare = categorical_columns[col]
            columns_compare_against = categorical_columns[(col+1):columns_length]
            chi_scores = chi2(train_data[columns_compare_against],train_data[column_to_compare])
            if column_to_compare not in columns_to_drop_cat:
                columns_to_be_dropped = [i for i in range(len(columns_compare_against)) if chi_scores[1][i]<=0.05]
                columns_to_drop_array = np.array(columns_compare_against)[columns_to_be_dropped]
                correlated_columns[column_to_compare]=columns_to_drop_array
                columns_to_drop_cat.update(columns_to_drop_array)


# In[41]:


train_data = train_data.drop(columns_to_drop_cat,axis=1)


# In[42]:


correlated_features = list(correlated_features)
columns_to_drop_cat = list(columns_to_drop_cat)
columns_to_drop_cat.extend(correlated_features)
columns_to_drop = columns_to_drop_cat.copy()


# ##### Visualizing the Output Variable

# In[43]:


sns.distplot(train_data['loss'], hist=True, kde=True,
bins=int(180/5), color = 'darkblue',
hist_kws={'edgecolor':'black'},
kde_kws={'linewidth': 4})


# In[44]:


train_data['loss'] = np.log(train_data['loss'])


# In[45]:


# Density Plot and Histogram of loss\n",
sns.distplot(train_data['loss'], hist=True, kde=True,
bins=int(180/5), color = 'darkblue',
hist_kws={'edgecolor':'black'},
kde_kws={'linewidth': 4})


# In[46]:


sns.distplot(np.exp(train_data['loss']), hist=True, kde=True,
bins=int(180/5), color = 'darkblue',
hist_kws={'edgecolor':'black'},
kde_kws={'linewidth': 4})


# ### Fit an ML Model

# In[47]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


# In[48]:


Column_datatypes= train_data.dtypes
Integer_columns = list(Column_datatypes.where(lambda x: x =="int64").dropna().index.values)
train_data[Integer_columns] = train_data[Integer_columns].astype('category',copy=False)
X,y = train_data.drop(['id','loss'],axis=1),train_data['loss']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=6)


# In[49]:


rf_base = RandomForestRegressor(n_estimators = 10, random_state = 6,oob_score = True)
rf_base.fit(X_train, y_train)


# In[50]:


pickle.dump(rf_base, open("basemodel_rf", 'wb'))


# In[51]:


basedmodel_rf = pickle.load(open("basemodel_rf", 'rb'))


# In[52]:


#compare the model accuracies
Y_test_predict_base = basedmodel_rf.predict(X_test)
print("Base model accuracy:",np.sqrt(mean_squared_error(y_test, Y_test_predict_base)))


# #### HyperParameter Tuning Using RandomSearchCV

# In[53]:


#number of trees
n_estimators = [100,200,300,400,500]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
min_samples_split = [200,400,600]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
   
    # Create the random grid
random_grid = {'n_estimators': n_estimators,
'max_features': max_features,
'max_depth': max_depth,
'min_samples_split': min_samples_split,
'min_samples_leaf': min_samples_leaf,
'bootstrap': bootstrap}


# In[54]:


rf = RandomForestRegressor()
rf_tuned = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 3,n_iter = 5, verbose=2, random_state=42, n_jobs = -1)
rf_tuned.fit(X_train, y_train)
pickle.dump(rf_tuned, open("tunedmodel_rf", 'wb'))


# In[59]:


rf_tuned.best_params_


# In[56]:


tunedmodel_rf = pickle.load(open("tunedmodel_rf", 'rb'))
Y_test_predict_tuned = tunedmodel_rf.predict(X_test)
print("Tuned model accuracy:",np.sqrt(mean_squared_error(y_test, Y_test_predict_tuned)))


#   ##### fit a GBM model

# In[57]:


from sklearn.ensemble import GradientBoostingRegressor
gbm_base = GradientBoostingRegressor(
max_depth=2,
n_estimators=3,
learning_rate=1.0)
gbm_base.fit(X_train,y_train)
pickle.dump(gbm_base, open("basemodel_GBM", 'wb'))


# In[58]:


basemodel_GBM = pickle.load(open("basemodel_GBM", 'rb'))
Y_test_predict_tuned = basemodel_GBM.predict(X_test)
print("Base model GBM accuracy:",np.sqrt(mean_squared_error(y_test, Y_test_predict_tuned)))

