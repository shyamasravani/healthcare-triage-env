import random
from typing import Dict, Any
from pydantic import BaseModel

class Patient(BaseModel):
    age: int
    hemoglobin: float
    blood_pressure: int
    symptoms: str

class State(BaseModel):
    patients: list
    current_index: int

class HealthcareTriageEnv:
    """
    Healthcare triage environment that generates patients and evaluates decisions.
    """

    def __init__(self):
        self.patients = []
        self.current_index = 0
        self.done = False

    def reset(self) -> Dict[str, Any]:
        # Generate a batch of patients
        self.patients = [
            Patient(
                age=random.randint(5, 80),
                hemoglobin=round(random.uniform(6, 15), 1),
                blood_pressure=random.randint(90, 160),
                symptoms=random.choice(["none", "fatigue", "dizziness", "chest pain"])
            ).dict()
            for _ in range(5)
        ]
        self.current_index = 0
        self.done = False
        return {"state": self.state().dict()}

    def step(self, action) -> Dict[str, Any]:
        if self.done:
            return {"reward": 0.0, "done": True, "state": self.state().dict()}

        patient = Patient(**self.patients[self.current_index])
        reward = self._evaluate_action(patient, action.decision)

        self.current_index += 1
        if self.current_index >= len(self.patients):
            self.done = True

        return {"reward": reward, "done": self.done, "state": self.state().dict()}

    def state(self) -> State:
        return State(patients=self.patients, current_index=self.current_index)

    def _evaluate_action(self, patient: Patient, decision: str) -> float:
        # Reward logic based on hemoglobin and decision
        if patient.hemoglobin < 8 and decision == "Refer":
            return 1.0
        elif patient.hemoglobin < 8 and decision != "Refer":
            return 0.2
        elif patient.hemoglobin >= 12 and decision == "Healthy":
            return 1.0
        elif decision == "RequestTest":
            return 0.3
        else:
            return 0.5

# Adapter class so inference.py can import MyEnvV4Env
class MyEnvV4Env(HealthcareTriageEnv):
    pass