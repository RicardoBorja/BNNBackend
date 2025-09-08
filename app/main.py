from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.controller import training, file_controller

app = FastAPI(
    title="XGBoost with FastAPI",
    description="API para entrenar y predecir con modelos XGBoost y ajuste bayesiano.",
    version="1.0.0",
)

origins = [
    "http://localhost:5173",  # local url
    "https://xgboostfrontend.vercel.app",  # frontend url
    "https://xgboostfrontend-git-bnn-impl-donjosh-uas-projects.vercel.app",  # frontend test url
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(training.router, prefix="/train", tags=["Entrenamiento"])
app.include_router(file_controller.router, prefix="/files", tags=["Files"])


@app.get("/", tags=["root"])
def root():
    return {"message": "Bienvenido a la API de BNN con FastAPI"}
