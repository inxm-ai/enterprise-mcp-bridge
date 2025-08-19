from fastapi import FastAPI
from .routes import router
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
instrumentator = Instrumentator()
app.include_router(router)
instrumentator.instrument(app).expose(app)
