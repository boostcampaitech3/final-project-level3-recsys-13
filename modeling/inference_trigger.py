from typing import Optional, Dict
import subprocess

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    config: Optional[Dict]
    

def command(model_name:str, config:dict):
    config_list = ['python','inference.py', model_name]
    for key, item in config.items():
        config_list.append(key)
        config_list.append(item)
    subprocess.run(
        config_list,
        cwd = '/opt/ml/final-project-level3-recsys-13/modeling/'
        )
    
    
app = FastAPI()

@app.post("/modeling")
def inference_model(model: Item):
    command(model.name, model.config)
    
    return True


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=30001)