import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="deploy_webhook", debug=False, version="0.1.0")

isAvailable = True

class Rate(BaseModel):
    rate : float

@app.post("api/v1/deploy_webhook", description="배포 시작 트리거")
async def get_orders(rate: Rate):
    global isAvailable
    if isAvailable:
        isAvailable = False
        return {"status": "success"}
    else:
        return {"status": "fail"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, debug=False)