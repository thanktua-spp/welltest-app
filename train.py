#@title ##Input the data
# Write short explanation @markdown
import pandas as pd
from utility import *
import io
from sklearn.model_selection import train_test_split
import pickle


# upload files
file_name = '/workspaces/welltest-app/Well 4SS x Inhouse Data.xlsx' #'/content/Well 4SS x Inhouse Data.xlsx'
welltest_df = get_data(file_name) #ok 'welltest/6LS Well Inhouse Data.xlsx'
welltest_df_stnd = standardize_column_names(welltest_df)
#print(welltest_df_stnd.head(3))

# prepare columns
model_columns = ['ds','Choke', 'FTHP', 'FLP','BS&W', 'OilRate']
welltest_new = data_processing(welltest_df_stnd, model_columns)
welltest_new['ds'] = pd.to_datetime(welltest_new['ds'])


# Prepare data and train model
drop_col = ['ds', 'y']
target='y'

# Generate dummy data
data = welltest_new.rename(columns={'OilRate': 'y'})
data = prepare_prophet_data(data)
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)

print(train_data.head(3))
models = train_models(train_data, drop_col, target)

decision_tree_model, rf_model, xgb_model, _ = models


# save models
import pickle
import os

# Assuming you have three pickle models named rf_model, xgb_model, and decision_tree_model
models_dic = {'random_forest': rf_model, 'xgboost': xgb_model, 'decision_tree': decision_tree_model}

# Create the 'models' folder if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the models as pickle files in the 'models' folder
for model_name, model in models_dic.items():
    with open(f'models/{model_name}.pkl', 'wb') as file:
        pickle.dump(model, file)

