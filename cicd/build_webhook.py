import uvicorn
from fastapi import FastAPI

app = FastAPI(title="build_webhook", debug=False, version="0.1.0")

isAvailable = True

@app.post("api/v1/build_webhook", description="빌드 시작 트리거")
async def get_orders():
    global isAvailable
    if isAvailable:
        isAvailable = False
        return {"status": "success"}
    else:
        return {"status": "fail"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False, debug=False)