#from metaflow import Flow
import pandas as pd
import os
import matplotlib.pyplot as plt

import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from prophet import Prophet

def slice_date(df, start_date, end_date):
    df['ds'] = pd.to_datetime(df['ds'])
    filtered_df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]
    return filtered_df

def get_data(data_path):
    """_summary_
    """
    #good keep
    return pd.read_excel(data_path, header=1)

def standardize_column_names(df):
    # good keep
    preferred_columns = [
    'Data Note', 'Well Name', 'Date', 'Time', 'Choke', 'FTHP', 'FTHT', 'FLP', 'Tsep', 'Psep',
    'Pmani', 'Meter Totalizer(Bbls)', 'Meter Factor', 'LiqRate', '%Water', '%Sediment',
    'BS&W', 'OilRate', 'DOF Plate size(inch)', 'GasDP(InchH20)', 'GasRate', 'GOR',
    'Sand(pptb)', 'Oil gravity (API)'
    ]
    
    existing_columns = df.columns.tolist()

    column_mapping = {}
    for i, existing_column in enumerate(existing_columns):
        if existing_column in preferred_columns:
            column_mapping[existing_column] = preferred_columns[i]
        else:
            column_mapping[existing_column] = None

    df.rename(columns=column_mapping, inplace=True)
    df = df[preferred_columns]

    return df


def upload_on_colab():
    uploaded = files.upload()
    for file_name, file_content in uploaded.items():
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content), header=1)
        elif file_name.endswith('.xls') or file_name.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(file_content), header=1)

        # Process the DataFrame
        df = standardize_column_names(df)

        # Display the DataFrame
        display(df)


def data_processing(welltest_df, model_columns):
    """_summary_
    """
    # checked and good
    welltest_df = standardize_column_names(welltest_df)
    welltest_df = standardize_column_names(welltest_df)
    welltest_drop_nan = welltest_df.dropna(axis=1).dropna(axis=0)
    X_temp = welltest_drop_nan.drop(['Well Name', 'Date', 'Time'], axis=1)
    X_temp = X_temp.apply(pd.to_numeric, errors='coerce')
    dropped_indices = X_temp[X_temp.isna().any(axis=1)].index
    X_temp = X_temp.dropna(axis=0)
    date = welltest_df['Date'].drop(dropped_indices).to_frame(name='ds')
    X = pd.concat([date, X_temp], axis=1)
    selected_columns = X.select_dtypes(include=['object']).columns
    if len(selected_columns) > 0:
        X_selected = X.astype(float)
        X_dropped = X.drop(selected_columns, axis=1)
        X = pd.concat([X_dropped, X_selected], axis=1)
    assert X.isna().sum().sum() == 0
    return X[model_columns]


# Prepare the data for Prophet model
def prepare_prophet_data(data):
    prophet_data = data.rename(columns={'OilRate': 'y'})
    return prophet_data

# Train the models
def train_models(data, drop_col, target):
    X_train = data.drop(drop_col, axis=1)#[features]
    y_train = data[target]
    # print('train data',X_train.columns)
    # Train Decision Tree model
    decision_tree_model = DecisionTreeRegressor(random_state=42)
    decision_tree_model.fit(X_train, y_train)

    # Train Random Forest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Train XGBoost model
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)

    # Train Prophet model
    prophet_model = Prophet()
    prophet_model.fit(data)

    return decision_tree_model, rf_model, xgb_model, prophet_model

# Make predictions using each model
def make_predictions(models, data, drop_col):
    X_test = data.drop(drop_col, axis=1)#[features]
    # print('test data', X_test.columns)
    svr_model, rf_model, xgb_model, prophet_model = models

    svr_preds = svr_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)

    prophet_preds = prophet_model.predict(data)['yhat']

    return svr_preds, rf_preds, xgb_preds, prophet_preds

# Calculate RMSE and R2 for each model
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def test_fit_plot(test_labels, test_predictions):
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    lims = [0, 2000]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    


def plot_compare_prediction(test_predictions, test_labels):
    import matplotlib.pyplot as plt

    plt.scatter(range(len(test_predictions)), test_predictions, c='blue', marker='o', label='Actual')
    plt.scatter(range(len(test_labels)), test_labels, c='red', marker='*', label='Predicted')

    # Adding labels and legend
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()

    # Display the plot
    plt.show()

def error_hist(test_predictions, test_labels):
    error = test_labels - test_predictions
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [MPG]')
    _ = plt.ylabel('Count')
