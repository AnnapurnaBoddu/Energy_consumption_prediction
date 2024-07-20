import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.stats import entropy
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import boxcox
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle


data = pd.read_csv('energy_consumption.csv')
print(data)
print(data.info())
print(data.describe())
print(data.isnull().sum())
# In cloud_coverage  column we have 304516 missing values (close to half data pounts), so we need to drop the column

data.drop(columns ='cloud_coverage',axis=1,inplace=True)

# using timestamp column we can extract the new features
data['year'] = pd.to_datetime(data['timestamp']).dt.year
data['month'] = pd.to_datetime(data['timestamp']).dt.month
data['day'] = pd.to_datetime(data['timestamp']).dt.day
data['hours'] = pd.to_datetime(data['timestamp']).dt.hour

# we extracted new features from timestamp column ,so we need to remove timestamp column from data set
data.drop(columns='timestamp',axis=1,inplace=True)
print(data)


# Replace negative values in 'precip_depth_1_hr' with NaN(bacuse the values should start from 0 not negative)
data['precip_depth_1_hr'] = data['precip_depth_1_hr'].apply(lambda x: x if x >= 0 else np.nan)


# we need to drop the year column beacuse it is constant column(all data points with constant value 2016)
data.drop(columns='year',axis=1,inplace=True)

print(data.isnull().sum())


numeric_cols = ['building_id', 'month', 'day', 'hours', 'square_feet', 'year_built', 'air_temperature',
                'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure','wind_direction', 'wind_speed']
categorical_cols = ['primary_use']
# Handling missing values using mean, meadian or mode
# for numerical columns with meadian
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())


# For categorical columns, fill missing values with the most frequent value
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

print(data.isnull().sum())


def plot(df, column):
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    sns.boxplot(data=df, x=column)
    plt.title(f'Box Plot for {column}')

    plt.subplot(1, 3, 2)
    sns.histplot(data=df, x=column, kde=True, bins=50)
    plt.title(f'Distribution Plot for {column}')

    plt.subplot(1, 3, 3)
    sns.violinplot(data=df, x=column)
    plt.title(f'Violin Plot for {column}')
    plt.show()

# observing the skewnwss in numeric_columns by using visualization
for col in numeric_cols:
  plot(data,col)

# claculating skewnwss in columns by using skew methon present in stats
skewed_feats = data[numeric_cols].apply(lambda x: skew(x.dropna()))
print(skewed_feats)

# handling skewness in columns
# positive skewed data transformed using log transformation
# negative skewed data transformes using square root transformation
df = data.copy()
# Transformations for highly skewed features
df['square_feet'] = np.log1p(df['square_feet'])  # Log transformation
df['precip_depth_1_hr'] = np.log1p(df['precip_depth_1_hr'])  # Log transformation

# Transformations for moderately skewed features
df['dew_temperature'] = np.square(df['dew_temperature'])  # Square transformation with abs to handle negative values

for col in numeric_cols:
  plot(df,col)

# claculating skewnwss in columns by using skew methon present in stats afterapllying log tranformation
skewed_feats = df[numeric_cols].apply(lambda x: skew(x.dropna()))
print(skewed_feats)


# mapping the categorical column
primary_use_map = {'Entertainment/public assembly': 0,'Lodging/residential': 1, 'Office': 2, 'Other': 3, 'Parking': 4, 'Retail': 5, 'Education': 6}
df['primary_use'] = df['primary_use'].map({'Entertainment/public assembly': 0,'Lodging/residential': 1, 'Office': 2, 'Other': 3, 'Parking': 4, 'Retail': 5, 'Education': 6})
'''
# Initialize OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False)  # drop='first' is used to avoid multicollinearity

# Fit and transform the data
encoded_features = encoder.fit_transform(df[['primary_use']])
with open('onehotencoder.pkl','wb') as f:
    pickle.dump(, f)


# Get the feature names for the encoded columns
encoded_feature_names = encoder.get_feature_names_out(['primary_use'])

# Create a DataFrame with the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
# Drop the original categorical column
df = df.drop('primary_use', axis=1)

# Concatenate the DataFrame with the encoded features
df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
'''

# Check the final DataFrame
print(df.head())

# from the above box plot observation hours,day, wind_speed,precip_depth_1_hr, dew_temparature, air_temparature, year_built, square_feet, primary_use are not distributed properly so we need apply log transformation on this columns

# outliers handling IQR method
# Using IQR and clip() methods to handle the outliers and add a new column of dataframe
df1 = df.copy()
def outlier(df, column):
    iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
    upper_threshold = df[column].quantile(0.75) + (1.5 * iqr)
    lower_threshold = df[column].quantile(0.25) - (1.5 * iqr)
    df[column] = df[column].clip(lower_threshold, upper_threshold)


outlier(df1,'wind_speed')
outlier(df1,'sea_level_pressure')
outlier(df1,'precip_depth_1_hr')
outlier(df1,'air_temperature')
outlier(df1,'square_feet')

for col in numeric_cols:
  plot(df1,col)


print(df1.isnull().sum())

def machine_learning_regression(df, algorithm):
    X = df1.drop('meter_reading',axis=1)
    print(X)
    print(X.columns())
    y = df1['meter_reading']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = algorithm().fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    # algo = str(algorithm).split("'")[1].split(".")[-1]
    accuracy_metrics = {'algorithm': algorithm.__name__,
                        'R2_train': r2_train,
                        'R2_test': r2_test}

    return accuracy_metrics


print(machine_learning_regression(df1, DecisionTreeRegressor))
print(machine_learning_regression(df1, ExtraTreesRegressor))
print(machine_learning_regression(df1, RandomForestRegressor))
print(machine_learning_regression(df1, AdaBoostRegressor))
print(machine_learning_regression(df1, GradientBoostingRegressor))
print(machine_learning_regression(df1, XGBRegressor))

param_grid_r = {'max_depth': [2, 5, 10, 20,30,40],
                'min_samples_split': [2, 5, 10,20],
                'min_samples_leaf': [1, 2, 4,6,7,10],
                'max_features': ['sqrt', 'log2', None]}

X = df1.drop('meter_reading',axis=1)
y = df1['meter_reading']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid_search_r = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid_r, cv=5, n_jobs=-1)
grid_search_r.fit(x_train, y_train)

# best parameters for hypertuning the random forest algorithm for better accuracy in unseen data
print(grid_search_r.best_params_, grid_search_r.best_score_)



# predict the selling price with hypertuning parameters and calculate the accuracy using metrics

X = df1.drop('meter_reading',axis=1)
y = df1['meter_reading']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X.columns())
print(X)

model = RandomForestRegressor(max_depth=40, max_features=None, min_samples_leaf=1, min_samples_split=2).fit(x_train,
                                                                                                            y_train)
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

metrics_r = {'R2': r2,
             'Mean Absolute Error': mae,
             'Mean Squared Error': mse,
             'Root Mean Squared Error': rmse}

print(metrics_r)



# save the regression model by using pickle

with open('regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)


