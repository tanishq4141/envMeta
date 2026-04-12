from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)
response = client.post("/reset")
print("POST /reset (no body):", response.status_code, response.text)

response2 = client.post("/reset", json={"task_name": "easy_clause_detection"})
print("POST /reset (with body):", response2.status_code, response2.text)
