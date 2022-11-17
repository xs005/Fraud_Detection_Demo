import warnings
from typing import Union, List

import joblib
import numpy as np
import pandas as pd
from pydantic import Json

warnings.filterwarnings('ignore')

pre_config = joblib.load('data/pre_config_data.joblib')


async def predict(x: Union[List[dict], dict], is_individual_call: bool) -> Json:
    if is_individual_call:
        x = [x]

    train = pd.DataFrame(x)
    all_train = await generate_features(train)
    model = pre_config['model']

    outcomes_train_final = pd.DataFrame(
        np.max(model.predict_proba(all_train), axis=1)).rename(columns={0: 'phat'})
    outcomes_train_final['business_outcome'] = (model.predict(all_train))
    train = pd.concat([train, outcomes_train_final], axis=1, sort=False)
    predictions = train.to_json(orient="records")

    if is_individual_call:
        # return individual for individual call
        predictions = predictions[1:-1]

    return predictions


async def generate_features(train: pd.DataFrame) -> pd.DataFrame:
    df = train.copy()
    high_freq_fraud_merchant_names = pre_config['high_freq_fraud_merchant_names']
    selected_features = pre_config['selected_features']

    # clean up
    # remove some useless columns
    useless_cols = ['echoBuffer', 'merchantCity', 'merchantState', 'merchantZip', 'posOnPremises', 'recurringAuthInd',
                    'accountNumber', 'customerId', 'cardLast4Digits']
    df.drop(columns=useless_cols, inplace=True)

    # create features related to CVV and remove the old CVV features
    df['correctedCVV'] = df.cardCVV == df.enteredCVV
    df.drop(columns=['cardCVV', 'enteredCVV'], inplace=True)

    # calculate some dates
    date_cols = ['transactionDateTime', 'currentExpDate', 'accountOpenDate', 'dateOfLastAddressChange']
    for i in date_cols:
        df[i] = pd.to_datetime(df[i])

    # create some date features
    df['transactionDays'] = (df.transactionDateTime.dt.date - df.accountOpenDate.dt.date).dt.days
    df['currentExpDays'] = (df.currentExpDate.dt.date - df.accountOpenDate.dt.date).dt.days
    df['addressChangeDays'] = (df.dateOfLastAddressChange.dt.date - df.accountOpenDate.dt.date).dt.days
    df['transactionDOW'] = df.transactionDateTime.dt.dayofweek
    df['transactionWeek'] = df.transactionDateTime.dt.isocalendar().week
    df['transactionHour'] = df.transactionDateTime.dt.hour

    # remove date_cols
    df.drop(columns=date_cols, inplace=True)

    # remove object columns with empty string, this is kind of null values
    for i in df.select_dtypes(include=['object']).columns:
        df = df.loc[df[i] != '']

    # clean up merchantName
    for i in high_freq_fraud_merchant_names:
        try:
            df.loc[df.merchantName.str.contains(f'{i} #'), 'merchantName'] = i.strip()
        except:
            pass
    # use others to replace other merchantName values
    df.loc[~df.merchantName.isin(high_freq_fraud_merchant_names), 'merchantName'] = 'others'

    # generate dummy features and make up all features
    abt = pd.get_dummies(df)
    missing_cols = list(set(selected_features) - set(abt.columns))
    if missing_cols:
        abt[missing_cols] = 0

    return abt[selected_features]
