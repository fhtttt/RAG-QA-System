from fastapi import FastAPI
from chat.chat import chat
import uvicorn

app = FastAPI(
    description="RAG Q&A Web API Server"
)

app.post("/api/chat",
         tags=["Chat"],
         summary="RAG Q&A Interface",
         )(chat)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000) # uvicorn is asgi server run fastapi