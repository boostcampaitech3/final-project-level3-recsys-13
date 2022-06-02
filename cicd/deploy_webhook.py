import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess

app = FastAPI(title="deploy_webhook", debug=False, version="0.1.0")

class GeneralRequest(BaseModel):
    branch_name : str

@app.post("/api/v1/deploy_webhook", description="배포 시작 트리거")
async def deploy_webhook(general: GeneralRequest):
    try:
        subprocess.call("chmod 777 ./deploy.sh", shell=True)
        result = subprocess.run(["./deploy.sh"], stdout=subprocess.PIPE, text=True, shell=True)
        print(result.stdout)
        print(general.branch_name)
        return {"result": "success"}
    except Exception as e:
        print(e)
        return {"result": "fail"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30003, reload=False, debug=True)