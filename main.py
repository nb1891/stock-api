
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from london_alpha_mvp import AutoData, compute_features, decide

app = FastAPI()

# gör så din webbapp kan prata med denna server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"status": "ok"}

def _get_company_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).get_info()
        return info.get("longName") or info.get("shortName") or info.get("displayName") or ticker
    except Exception:
        return ticker  # fallback om inget namn hittas

@app.get("/api/evaluate")
def evaluate(ticker: str = Query(..., description="e.g. AAPL or HSBA.L"),
             horizon_m: int = Query(6, ge=1, le=36),
             explain: bool = False):
    try:
        auto = AutoData(ticker)
        prices, bench = auto.fetch()
        feats = compute_features(prices, bench, horizon_m=horizon_m)
        decision = decide(feats, horizon_m=horizon_m)

        return {
            "ticker": ticker,
            "horizon_m": horizon_m,
            "decision": decision
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



