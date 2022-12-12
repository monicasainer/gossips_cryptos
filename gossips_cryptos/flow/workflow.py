from gossips_cryptos.interface.main import preprocess, train ,pred_new_day

from prefect import task, Flow, Parameter
from prefect.schedules import IntervalSchedule
from datetime import timedelta
import os
import requests

#Create function/task to evaluate.
@task
def predict():
    new_day = pred_new_day()
    eval_mae = abs(new_day['predicted_price'][0]-new_day['actual_price'].item())
    return eval_mae


@task
def train_production_model():
    """
    Run the `Production` stage evaluation on new data
    Returns `eval_mae`
    """
    # $CHA_BEGIN
    train_mae = train()
    return train_mae.item()

@task
def notify(eval_mae, train_mae):
    base_url = 'https://wagon-chat.herokuapp.com'
    channel = 'gossips_cryptos'
    url = f"{base_url}/{channel}/messages"
    author = 'monica'
    content = "Evaluation MAE: {} - New training MAE: {}".format(
        round(eval_mae, 2), round(train_mae, 2))
    data = dict(author=author, content=content)
    response = requests.post(url, data=data)
    response.raise_for_status()

def build_flow():
    """
    build the prefect workflow for the `gossips_cryptos` package
    """
    # $CHA_BEGIN
    flow_name = os.environ.get("PREFECT_FLOW_NAME")


    schedule = IntervalSchedule(interval=timedelta(days=1))

    with Flow(flow_name, schedule=schedule) as flow:

        # retrieve mlfow env params
        #mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")

        # create workflow parameters
        #experiment = Parameter(name="experiment", default=mlflow_experiment)

        # register tasks in the workflow
        eval_mae = predict()
        train_mae = train_production_model()
        notify(eval_mae,train_mae)

    return flow
    # $CHA_END

#flow.run()
