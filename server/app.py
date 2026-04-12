from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class StepInput(BaseModel):
    decision: str

state = {"current_decision": None}

@app.post("/reset")
def reset():
    state["current_decision"] = None
    return {"status": "reset"}

@app.post("/step")
def step(input: StepInput):
    decision = input.decision
    if decision not in ["Refer", "Treat", "Discharge"]:
        raise HTTPException(status_code=400, detail="Invalid decision")
    state["current_decision"] = decision
    return {"status": "step", "decision": decision}

@app.get("/state")
def get_state():
    return {"status": "state", "current_decision": state["current_decision"]}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
