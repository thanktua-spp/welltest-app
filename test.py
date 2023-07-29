import pickle
import pandas as pd
import numpy as np

# Assuming you have three model names: 'random_forest', 'xgboost', and 'decision_tree'
model_names = ['random_forest', 'xgboost', 'decision_tree']

# Load the models into a tuple of model variables
loaded_models = tuple()

for model_name in model_names:
    with open(f'models/{model_name}.pkl', 'rb') as file:
        model = pickle.load(file)
        loaded_models += (model,)

#print(loaded_models)
random_forest_model, xgboost_model, decision_tree_model = loaded_models

model_columns = ['Choke', 'FTHP', 'FLP','BS&W']
test_data_list = [32, 1000, 280, 0.45]
data_dict = {col: [test_data_list[i]] for i, col in enumerate(model_columns)}
test_df = pd.DataFrame(data_dict)
#print(test_data)

print(loaded_models[0].predict(test_df),
      loaded_models[1].predict(test_df),
      loaded_models[2].predict(test_df)
      )




# Function to make predictions using each model in the loaded_models tuple
# def make_predictions(models, data):
#     predictions = []
#     for model in models:
#         # Assuming the model has a predict method. Replace it with the actual method name if different.
#         prediction = model.predict([data])  # Passing the numerical features [32, 1000, 280, 0.45, 775.12]
#         predictions.append(prediction[0])
#     return predictions

# # Call the function to make predictions using the test data and loaded_models
# predictions = make_predictions(loaded_models, test_data)

# # Print the predictions for each model
# for model_name, prediction in zip(model_names, predictions):
#     print(f"Prediction using {model_name}: {prediction}")
