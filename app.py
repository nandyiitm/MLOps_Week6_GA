
# use fastapi and model.pkl for making predictions given a new data point in dict format with keys sepal_length,sepal_width,petal_length,petal_width.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('model.pkl')

app = FastAPI(title="Iris Classifier")

# Define the expected input format
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
async def predict_species(data: IrisData):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])

        # Make prediction
        prediction = model.predict(df)[0]

        return {"predicted_species": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

