import pandas as pd
import missingno as msno
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import seaborn as sns
from sklearn.model_selection import train_test_split
import category_encoders as ce 
import numpy as np
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
#Logan Buddenbaum certified Code

#Handled in the script for dashboard

#Functions/graphs that show the missing values within the data
def missingvalsVisuals(df):
    fig, ax = plt.subplots(figsize=(12,8))
    msno.bar(df, ax=ax)
    plt.title(f'Missing Values Bar Plot')
    plt.subplots_adjust(bottom=-.2)
    #Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    #Convert image
    img = np.array(Image.open(buf))
    plotly_fig = go.Figure(go.Image(z=img))
    plotly_fig.update_layout(width=1200, height=960)
    return plotly_fig

def summarystatsbymethods(df):
    head_df = df.head()
    describe_df = df.describe(include='all')
    describe_df = describe_df.reset_index()
    return head_df, describe_df

def remove_rows_with_missing_values(df):
    """Removes all rows with any missing values"""
    clean_df = df.dropna()
    return clean_df


def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = df[column_name].between(lower_bound, upper_bound, inclusive='both')
    return df[mask], df[~mask]


def plot_histogramv2(df, column, bins=20, xlim=None, ylim=None):
    #Create histogram with Matplotlib
    plt.figure(figsize=(20,12))
    sns.histplot(df[column], bins=bins)
    plt.title(f'Histogram of {column}')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.grid()
    #Save to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    #load image from buffer
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    #Create Plotly figure
    fig = go.Figure(go.Image(z=img_array))
    fig.update_layout(width=1200, height=960)
    return fig


def plot_barchart(df, column, title=None):
    """Plots a bar chart for the categorical data to visualize trends"""
    value_counts = df[column].value_counts()
    plt.figure(figsize=(10,6))
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title(title if title else f'Frequency of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.grid(axis='y', alpha=0.75)
    #Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    #Load image from buffer
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    #Convert numpy array to plotly figure
    fig = go.Figure(go.Image(z=img_array))
    fig.update_layout(width=1200, height=960)
    return fig

def correlation_matrix(df, columns):
    corr = df[columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    #Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    #Load image from buffer
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    #Convert numpy array to plotly figure
    fig = go.Figure(go.Image(z=img_array))
    fig.update_layout(width=1200, height=960)
    return fig


def prepare_date_features(df, date_column=None):
    df1 = df.copy()
    df1[date_column] = pd.to_datetime(df1[date_column])
    df1['year'] = df1[date_column].dt.year
    df1['month'] = df1[date_column].dt.month
    df1['day'] = df1[date_column].dt.day
    return df1

def plot_price_trends(df, date_column=None, price_column=None):
    df = df.sort_values(by=date_column)
    df.set_index(date_column, inplace=True)
    monthly_median_prices = df[price_column].resample('ME').median()
    plt.figure(figsize=(10,5))
    monthly_median_prices.plot(title='Median Price Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Median Price')
    plt.grid(True)
    #Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    #Load image from buffer
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    #Convert numpy array to plotly figure
    fig = go.Figure(go.Image(z=img_array))
    fig.update_layout(width=1200, height=960)
    return fig

def plot_location_price_effects(df, location_column=None, price_column=None):
    num_locations = df[location_column].nunique()
    fig, ax = plt.subplots(figsize=(12, max(6, 0.5 * num_locations)))
    box_plot = df.boxplot(column=price_column, by=location_column, vert=False, ax=ax)
    ax.set_ylim(0, num_locations + 1)
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.05)
    plt.tight_layout()
    plt.title('Price Distribution by ' + location_column)
    plt.xlabel('Price')
    plt.ylabel(location_column)
    plt.suptitle('')
    #Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    #Load image from buffer
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    #Convert numpy array to plotly figure
    fig = go.Figure(go.Image(z=img_array))
    fig.update_layout(width=2400, height=1920)
    return fig

def plot_size_vs_price(df, size_column=None, price_column=None):
    plt.figure(figsize=(8,6))
    plt.scatter(df[size_column], df[price_column], alpha=0.5)
    plt.title('Relationship between Property Size and Price')
    plt.xlabel('Property Size')
    plt.ylabel('Price')
    plt.grid(True)
    #Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    #Load image from buffer
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    #Convert numpy array to plotly figure
    fig = go.Figure(go.Image(z=img_array))
    fig.update_layout(width=1200, height=960)
    return fig

def plot_scatter(df, x_column, y_column):
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x=x_column, y=y_column)
    plt.title(f'Scatter Plot of {x_column} vs. {y_column}')
    #Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    #Load image from buffer
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    #Convert numpy array to plotly figure
    fig = go.Figure(go.Image(z=img_array))
    fig.update_layout(width=1200, height=960)
    return fig

def skewkurtfunction(df):
    data = df[['price', 'bed', 'bath', 'acre_lot', 'house_size', 'year', 'month', 'day']]
    #For skewness and kurtosis
    skewness = data.skew().reset_index()
    skewness.columns = ['Variable', 'Skewness']

    kurtosis = data.kurt().reset_index()
    kurtosis.columns = ['Variable', 'Kurtosis']
    return skewness, kurtosis
    

def prepare_and_split_data(df, test_size=0.2, random_state=42):
    """Prepares the data for modeling by selecting relevant features and splitting into train and test sets.

    Parameters:
    df (DataFrame): The input DataFrame that contains all the data including the target variable 'price'.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed used by the random number generator for reproducibility.

    Returns:
    X_train (DataFrame): Training features.
    X_test (DataFrame): Testing features.
    Y_train (Series): Training target variable.
    Y_test (Series): Testing target variable."""
    #Create the final cleaned DataFrame that contains features for the model
    final_features = df[['price', 'bed', 'bath', 'acre_lot', 'city', 'state', 'zip_code', 'house_size', 'year', 'month', 'day']]
    #Split the data into features and target
    X = final_features.drop('price', axis=1)
    Y = final_features['price']
    #Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test

def apply_multi_level_target_encoding(X_train, X_test, Y_train, columns):
    """Applies multi-level target encoding to combined location features and cleans up the original columns.

    Parameters:
    X_train (DataFrame): Training feature data.
    X_test (DataFrame): Testing feature data.
    Y_train (Series): Training target data.
    columns (list): List of column names to be combined and encoded.

    Returns:
    X_train (DataFrame): Modified training data with encoded features.
    X_test (DataFrame): Modified testing data with encoded features."""
    #Combine specified columns into a single feature for encoding
    location_train = X_train[columns].astype(str).agg('_'.join, axis=1)
    location_test = X_test[columns].astype(str).agg('_'.join, axis=1)
    #Initialize and apply the target encoder
    encoder = ce.TargetEncoder()
    X_train['location_encoded'] = encoder.fit_transform(location_train, Y_train)
    X_test['location_encoded'] = encoder.transform(location_test)
    #Drop the original columns to avoid multicollinearity
    X_train = X_train.drop(columns, axis=1)
    X_test = X_test.drop(columns, axis=1)

    return X_train, X_test

def train_evaluate_LinearRegression(X_train, X_test, Y_train, Y_test):
    """Trains a linear regression model on the training data, makes predictions on the test set,
    and evaluates the model using Mean Squared Error (MSE) and R-squared (R2) metrics.

    Parameters:
    X_train (DataFrame): Training data features.
    Y_train (Series): Training data target.
    X_test (DataFrame): Testing data features.
    Y_test (Series): Testing data target.

    Returns:
    dict: A dictionary containing the model, predictions, and performance metrics."""
    #Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, Y_train)
    #Predict on the testing set
    Y_pred = model.predict(X_test)
    #Evaluate the model
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    #Print performance metrics
    #print(f"R^2 Score: {r2}")
    return {
        'model': model,
        'predictions': Y_pred,
        'mse': mse,
        'r2': r2
    }

def apply_cyclical_encoding(df_train, df_test, day_column='day', month_column='month'):
    """Applies cyclical encoding to 'day' and 'month' columns and removes the original columns.

    Parameters:
    df_train (DataFrame): Training dataset that contains 'day' and 'month' columns.
    df_test (DataFrame): Testing dataset that contains 'day' and 'month' columns.
    day_column (str): Column name for the day of the month.
    month_column (str): Column name for the month of the year.

    Returns:
    df_train (DataFrame): Modified training dataset with cyclical encoded day and month, and original columns removed.
    df_test (DataFrame): Modified testing dataset with cyclical encoded day and month, and original columns removed."""
    #Cyclical encoding for 'day'
    df_train[f'{day_column}_sine'] = np.sin(2 * np.pi * df_train[day_column] / 31)
    df_train[f'{day_column}_cosine'] = np.cos(2 * np.pi * df_train[day_column] / 31)
    df_test[f'{day_column}_sine'] = np.sin(2 * np.pi * df_test[day_column] / 31)
    df_test[f'{day_column}_cosine'] = np.cos(2 * np.pi * df_test[day_column] / 31)
    #Cyclical encoding for 'month'
    df_train[f'{month_column}_sine'] = np.sin(2 * np.pi * df_train[month_column] / 12)
    df_train[f'{month_column}_cosine'] = np.cos(2 * np.pi * df_train[month_column] / 12)
    df_test[f'{month_column}_sine'] = np.sin(2 * np.pi * df_test[month_column] / 12)
    df_test[f'{month_column}_cosine'] = np.cos(2 * np.pi * df_test[month_column] / 12)
    #Remove the original 'day' and 'month' columns
    df_train.drop([day_column, month_column], axis=1, inplace=True)
    df_test.drop([day_column, month_column], axis=1, inplace=True)

    return df_train, df_test

def train_evaluate_randomforest(X_train, X_test, Y_train, Y_test, n_estimators=100, random_state=42):
   """Trains a Random Forest regressor on the training data, makes predictions on the test set,
    and evaluates the model using Mean Squared Error (MSE) and R-squared (R2) metrics.

    Parameters:
    X_train (DataFrame): Training data features.
    Y_train (Series): Training data target.
    X_test (DataFrame): Testing data features.
    Y_test (Series): Testing data target.
    n_estimators (int): The number of trees in the forest.
    random_state (int): Controls both the randomness of the bootstrapping of the samples used
                        when building trees (if bootstrap=True) and the sampling of the features to consider when
                        looking for the best split at each node.

    Returns:
    dict: A dictionary containing the model, predictions, and performance metrics."""
   #Initialize the Random Forest model
   rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
   #Train the model
   rf_model.fit(X_train, Y_train)
   #Predict on the test set
   rf_predictions = rf_model.predict(X_test)
   #Evaluate the model performance
   rf_mse = mean_squared_error(Y_test, rf_predictions)
   rf_r2 = r2_score(Y_test, rf_predictions)
   #Print the performance metrics
   #print("Random Forest MSE:", rf_mse)
   #print("Random Forest R^2:", rf_r2)

   return {
      'model': rf_model,
      'predictions': rf_predictions,
      'mse': rf_mse,
      'r2': rf_r2
   }

def train_evaluate_gbm(X_train, X_test, Y_train, Y_test, n_estimators=100, learning_rate=0.1, random_state=42):
    """Trains a Gradient Boosting regressor on the training data, makes predictions on the test set,
    and evaluates the model using Mean Squared Error (MSE) and R-squared (R2) metrics.

    Parameters:
    X_train (DataFrame): Training data features.
    Y_train (Series): Training data target.
    X_test (DataFrame): Testing data features.
    Y_test (Series): Testing data target.
    n_estimators (int): The number of boosting stages to perform.
    learning_rate (float): Learning rate shrinks the contribution of each tree.
    random_state (int): Controls the random seed given at each base learner iteration.

    Returns:
    dict: A dictionary containing the model, predictions, and performance metrics."""
    #Initialize the Gradient Boosting Regressor
    gbm_model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    #Train the model
    gbm_model.fit(X_train, Y_train)
    #Predict on the test set
    gbm_predictions = gbm_model.predict(X_test)
    #Evaluate the model performance
    gbm_mse = mean_squared_error(Y_test, gbm_predictions)
    gbm_r2 = r2_score(Y_test, gbm_predictions)
    #Print the performance metrics
    #print("Gradient Boosting MSE:", gbm_mse)
    #print("Gradient Boosting R^2:", gbm_r2)
    return {
        'model': gbm_model,
        'predictions': gbm_predictions,
        'mse': gbm_mse,
        'r2': gbm_r2
    }
