import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import subprocess

app = FastAPI(title="deploy_webhook", debug=False, version="0.1.0")

isAvailable = True

class Rate(BaseModel):
    rate : str

@app.post("/api/v1/deploy_webhook", description="배포 시작 트리거")
async def get_orders(rate: Rate):
    global isAvailable
    subprocess.call("chmod 777 ./deploy.sh", shell=True)
    result = subprocess.run(["./deploy.sh"], stdout=subprocess.PIPE, text=True, shell=True)
    print(result.stdout)
    if isAvailable:
        isAvailable = False
        return {"status": "success"}
    else:
        return {"status": "fail"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30003, reload=False, debug=True)