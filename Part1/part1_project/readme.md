![alt text](../imgs/architecture.png)

```
python server/api_router.py
```

```
curl --location 'http://127.0.0.1:8000/api/chat' \
--header 'Content-Type: application/json' \
--data '{
    "query": "please introduce yourself",
    "model_name": "glm-4",
    "temperature": 0.8,
    "max_tokens": 4096
}'
```