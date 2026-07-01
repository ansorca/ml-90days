
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import numpy as np
from stage1.bootstrap import create_classifier
import os
from stage1.adapters.http.schemas import PredictRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier_name = os.getenv("CLASSIFIER", "hybrid")
    (classifier, class_names) = create_classifier(classifier_name)
    app.state.model = classifier
    app.state.class_names = class_names
   
    yield


app = FastAPI(lifespan=lifespan)

@app.get("/health/live")
def liveness():
    return  {"status": "OK"}

@app.get("/health/ready")
def ready():
    if app.state.model and app.state.class_names.all():
        return {"status": "OK"}
    
    raise HTTPException(status_code=503, detail="Service unavailable")

@app.post("/predict/")
def predict_item(features: PredictRequest):
    predictions = app.state.model.predict(np.array(features.data, dtype=np.float32))
    return {"data": [app.state.class_names[i] for i in predictions]}