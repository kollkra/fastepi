from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import logging
import uvicorn
from sklearn.preprocessing import OrdinalEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # Исправлено: было name

app = FastAPI(title="Car Price")

# Загрузка моделей
try:
    with open("models/cars.joblib", "rb") as f:
        model = pickle.load(f)
    with open("models/power.joblib", "rb") as f:
        predict2price = pickle.load(f)
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    predict2price = None

def clear_data(df):
    cat_columns = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
    # Добавлена обработка неизвестных значений, чтобы не падало
    ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ordinal.fit(df[cat_columns])
    encoded = ordinal.transform(df[cat_columns])
    df[cat_columns] = pd.DataFrame(encoded, columns=cat_columns)
    return df

def featurize(dframe):
    df = dframe.copy()
    # Защита от деления на ноль
    df['Distance_by_year'] = df['Distance'] / (2022 - df['Year']).replace(0, 1)
    df['age'] = 2024 - df['Year']
    
    if 'Style' in df.columns and 'Engine_capacity' in df.columns:
        mean_cap = df.groupby('Style')['Engine_capacity'].transform('mean')
        df['eng_cap_diff'] = (df['Engine_capacity'] - mean_cap).abs()
        max_cap = df.groupby('Style')['Engine_capacity'].transform('max')
        df['eng_cap_diff_max'] = (df['Engine_capacity'] - max_cap).abs()
    return df

class CarFeatures(BaseModel):
    make: str
    model: str
    year: int
    style: str
    distance: float
    engine_capacity: float
    fuel_type: str
    transmission: str

@app.post("/predict", summary="Predict car price")  # Исправлено: убраны пробелы
async def predict(car: CarFeatures):
    if model is None or predict2price is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    try:
        # Исправлено: убраны пробелы в названиях колонок
        columns_names = ["Make", "Model", "Year", "Style", "Distance", "Engine_capacity", "Fuel_type", "Transmission"]
        
        # Pydantic V2: model_dump() вместо dict()
        input_data = pd.DataFrame([car.model_dump()])
        input_data.columns = columns_names
        
        df_clean = clear_data(input_data)
        df_feat = featurize(df_clean)
        
        # Исправлено: опечатка в имени переменной
        pred = model.predict(df_feat)[0]
        price = predict2price.inverse_transform(pred.reshape(-1, 1))[0][0]
        
        return {"predicted_price": round(float(price), 2)}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":  # Исправлено: было name == "main"
    uvicorn.run(app, host="0.0.0.0", port=8005)
