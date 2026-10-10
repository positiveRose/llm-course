import requests

BASE = "http://127.0.0.1:8000"

resp = requests.post(BASE + "/sessions", headers={"x-user-id": "user1"})
session_id = resp.json()["session_id"]

resp = requests.post(BASE + "/sessions/" + session_id + "/messages", headers={"x-user-id": "user1"}, json={"text": "какой статус заказа прямо сейчас у 54010?"})
print(resp.text)
