from typing import Union, List

import uvicorn
from fastapi import FastAPI, Body
from pydantic import Json

from data.examples import individual_call
from utils import predict

app = FastAPI()


@app.post("/predict", response_model=Json, tags=['Prediction'])
async def get_prediction(x: Union[List[dict], dict] = Body(example=individual_call)):
    """
    Call function to predict. Input should be Json str, either single element or a list of elements.
    The output is Json format of the predicted outcome (variable name is business_outcome),
    predicted probability (variable name is phat), and all model inputs.
    """
    # check if input has all features
    if isinstance(x, dict):
        # individual call
        is_individual_call = True
    else:
        # batch call, it's less likely that a list of inputs have different features, just check the first one
        is_individual_call = False

    predictions = await predict(x, is_individual_call)
    return predictions


if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=1313, log_level='warning', reload=False)
