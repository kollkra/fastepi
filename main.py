from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import uvicorn
from sklearn.preprocessing import OrdinalEncoder
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Car Price")

# Загрузка моделей
model = joblib.load("models/cars.joblib")
predict2price = joblib.load("models/power.joblib")

# База данных
DATABASE_URL = "postgresql://user:password@db:5432/carprice"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Prediction(BaseModel):
    id: int
    make: str
    model: str
    year: int
    style: str
    distance: float
    engine_capacity: float
    fuel_type: str
    transmission: str
    predicted_price: float
    created_at: datetime

    class Config:
        from_attributes = True

class CarInput(BaseModel):
    make: str
    model: str
    year: int
    style: str
    distance: float
    engine_capacity: float
    fuel_type: str
    transmission: str

class PredictionDB(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    make = Column(String)
    model = Column(String)
    year = Column(Integer)
    style = Column(String)
    distance = Column(Float)
    engine_capacity = Column(Float)
    fuel_type = Column(String)
    transmission = Column(String)
    predicted_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

def clear_data(df, encoder=None, fit=False):
    cat_columns = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
    if fit:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoder.fit(df[cat_columns])
    encoded = encoder.transform(df[cat_columns])
    df[cat_columns] = pd.DataFrame(encoded, columns=cat_columns)
    return df, encoder

def featurize(df):
    df = df.copy()
    df['Distance_by_year'] = df['Distance'] / (2022 - df['Year']).replace(0, 1)
    df['age'] = 2024 - df['Year']
    if 'Style' in df.columns and 'Engine_capacity' in df.columns:
        mean_cap = df.groupby('Style')['Engine_capacity'].transform('mean')
        df['eng_cap_diff'] = (df['Engine_capacity'] - mean_cap).abs()
        max_cap = df.groupby('Style')['Engine_capacity'].transform('max')
        df['eng_cap_diff_max'] = (df['Engine_capacity'] - max_cap).abs()
    return df

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

@app.post("/predict")
async def predict(car: CarInput):
    try:
        cols = ["Make", "Model", "Year", "Style", "Distance", "Engine_capacity", "Fuel_type", "Transmission"]
        df = pd.DataFrame([{
            "Make": car.make, "Model": car.model, "Year": car.year, "Style": car.style,
            "Distance": car.distance, "Engine_capacity": car.engine_capacity,
            "Fuel_type": car.fuel_type, "Transmission": car.transmission
        }])
        df, _ = clear_data(df[cols], fit=False)
        df = featurize(df)
        pred = model.predict(df)[0]
        price = float(predict2price.inverse_transform([[pred]])[0][0])
        
        # Сохраняем в БД
        db = SessionLocal()
        record = PredictionDB(
            make=car.make, model=car.model, year=car.year, style=car.style,
            distance=car.distance, engine_capacity=car.engine_capacity,
            fuel_type=car.fuel_type, transmission=car.transmission,
            predicted_price=round(price, 2)
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        db.close()
        
        return {"predicted_price": round(price, 2)}
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def history(limit: int = 10):
    db = SessionLocal()
    records = db.query(PredictionDB).order_by(PredictionDB.created_at.desc()).limit(limit).all()
    db.close()
    return [{"predicted_price": r.predicted_price, "created_at": r.created_at} for r in records]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
