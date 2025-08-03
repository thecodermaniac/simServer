## 🚀 How to Setup

For Cuda(gpu) based systems

```bash
pip install -r requirements-gpu.txt
```

For cpu based systems

```bash
pip install -r requirements-cpu.txt
```

## The models are not uploaded to Github.

## Instead they are uploaded in drive. Download them and use them before running server

## How to set up the model

1. Download the zip file and extract the models. Can choose between GPU based or CPU based models.
2. create the folder "modelsAI" in the parent directory. Put the models under "modelsAI" folder.
3. If the models are downloaded with a different name rename them. For GPU:- tinyllama-merged-v3-gpu and for CPU:- tinyllama-merged-v3-cpu
4. Change to "gpu" / "cpu" based on pytorch setup in predictor.py file.

## Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Sample API request

Endpoint:- http://127.0.0.1:8000/predict

```bash
{
  "symbol": "ETH",
  "history": [150, 148.2, 147.5, 120.0, 98.3],
  "prompt": "Unexpected crash due to regulatory news!"
}
```

## Response

```bash
{
  "predicted_price": [
    127.6,
    116.2,
    102.2,
    84.4,
    56.8
  ]
}
```
