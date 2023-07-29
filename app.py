import warnings
warnings.filterwarnings("ignore")
import pickle
import pandas as pd
import numpy as np
import gradio as gr
import random

def import_model():
    model_names = ['random_forest', 'xgboost', 'decision_tree']
    loaded_models = tuple()
    for model_name in model_names:
        with open(f'models/{model_name}.pkl', 'rb') as file:
            model = pickle.load(file)
            loaded_models += (model,)
    return loaded_models

def predict_liquid_rate(*input):
    input_list = list(input)
    inp_arr = np.array(input_list[:-1]).reshape(1, -1)
    random_forest_model, xgboost_model, decision_tree_model = import_model()
    model_selection = input_list[-1]
    print(model_selection)

    result = {}
    model_names = {
        'XGBoost': 'XGBoost Oil Rate',
        'Random Forest': 'Random Forest Oil Rate',
        'Decision Tree': 'Decision Tree Oil Rate',
        'Prophet Model': 'Prophet Model Oil Rate'
    }
    xg_output = ''
    dt_output = ''
    rf_output = ''
    for choice in model_selection:
        if choice == 'XGBoost':
            xg_pred = xgboost_model.predict(inp_arr)
            result[choice] = xg_pred[0]
            xg_output = f"{model_names['XGBoost']}: {result['XGBoost']:.2f} Bbls/day"
        elif choice == 'Decision Tree':
            dt_pred = decision_tree_model.predict(inp_arr)
            result[choice] = dt_pred[0]
            dt_output = f"{model_names['Decision Tree']}: {result['Decision Tree']:.2f} Bbls/day"
        elif choice == 'Random Forest':
            rf_pred = random_forest_model.predict(inp_arr)
            result[choice] = rf_pred[0]
            rf_output = f"{model_names['Random Forest']}: {result['Random Forest']:.2f} Bbls/day"

    return xg_output, dt_output, rf_output

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Oil Rate Prediction
    Use this table as Reference for Last Well test data.
    """)
    with gr.Column():
        with gr.Box():
            frame_output = gr.Dataframe(
                value=[['2022-12-23', 32, 1000, 280, 0.45,  775.12]],
                headers=['Date', 'Choke', 'FTHP', 'FLP', 'BS&W', 'OilRate'],
                datatype=["str", "number", "number", "number",  "number", "number"],
                )

        gr.Markdown(
        """    Use the different input slider to select new welltest information
        """)
        with gr.Box():
            choke = gr.Slider(minimum=0, maximum=100, value=32, step=2, label="Choke Size (1/64\")", interactive=True)
            fthp = gr.Slider(minimum=500, maximum=5000, step=1, value=1000, label="Tubing Head Pressure (FTHP)(psi)", interactive=True)
            flp = gr.Slider(minimum=0, maximum=5000, step=1, value=280,label="Flow Line Pressure (FLP)(psi)", interactive=True)
            bsw = gr.Slider(minimum=0, maximum=100, value=0.45, label="Basic Sediment and Water (BS&W)(%)", interactive=True)

        gr.Markdown(
        """    Use the different trained models to perform Oil rate prediction
        """)
        # Output Controls
        with gr.Column():
            select_model = gr.CheckboxGroup(choices=["Random Forest", "XGBoost", "Decision Tree"], value='XGBoost', label="Select Model", info="Select Model to make prediction", interactive=True)
            btn_predict = gr.Button("Test Prediction")
            xg_output = gr.Label(label="XGBoost model")
            dt_output = gr.Label(label="Decision Tree")
            rf_output = gr.Label(label="Random Forest")


    input_items = [choke, fthp, flp, bsw, select_model]
    btn_predict.click(fn=predict_liquid_rate, inputs=input_items, outputs=[xg_output,dt_output,rf_output])
    #gr.describe()
    
demo.launch(debug=True, share=True)