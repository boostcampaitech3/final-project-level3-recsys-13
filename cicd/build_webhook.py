import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import requests

app = FastAPI(title="build_webhook", debug=False, version="0.1.0")

class GeneralRequest(BaseModel):
    data : str

@app.post("api/v1/build_webhook", description="빌드 시작 트리거")
async def build_webhook(general: GeneralRequest):
    try:
        branch_name = general.data
        subprocess.call(f"git checkout {branch_name}", shell=True)
        subprocess.call("git pull", shell=True)
        subprocess.call("chmod 777 ./build.sh", shell=True)
        result = subprocess.run(["./build.sh"], stdout=subprocess.PIPE, text=True, shell=True)
        requests.post("http://34.64.137.90:30003/api/v1/deploy_webhook", json={"data":'go'}).json()
        print(result.stdout)
        print(branch_name)
        return {"result": "success"}
    except Exception as e:
        print(e)
        return {"result": "fail"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30004, reload=False, debug=False)