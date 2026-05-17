import requests
import json

payload = {
    "config": {"resolution": "800p", "burst_mode": True},
    "p1": {"x": 1, "y": 1},
    "p2": {"x": 2, "y": 2},
    "actual": 100
}
res = requests.post("http://localhost:8080/test_config", json=payload)
print(res.status_code)
print(res.json())
