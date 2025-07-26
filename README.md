## 🚀 How to Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```


## Sample API request

Endpoint:- http://127.0.0.1:8000/predict

``` bash
{
  "symbol": "ETH",
  "history": [100, 88.5, 98.7, 101.3, 97.8],
  "admin_prompt": "Whales are buying in bulk due to government interest."
}
```

