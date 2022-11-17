from fastapi.testclient import TestClient

from data.examples import individual_call, individual_response, batch_call, batch_response
from main import app

client = TestClient(app, raise_server_exceptions=True)


def test_individual_call():
    response = client.post(
        "/predict",
        headers={"Content-Type": "application/json"},
        json=individual_call,
    )
    assert response.status_code == 200
    assert response.json() == individual_response


def test_batch_call():
    response = client.post(
        "/predict",
        headers={"Content-Type": "application/json"},
        json=batch_call,
    )
    assert response.status_code == 200
    assert response.json() == batch_response
