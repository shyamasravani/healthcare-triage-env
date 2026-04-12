from pydantic import BaseModel

class MyEnvV4Action(BaseModel):
    """
    Action model for the triage environment.
    Wraps a decision string such as: "Healthy", "Monitor", "Refer", "RequestTest".
    """
    decision: str

    def __repr__(self):
        return f"MyEnvV4Action(decision={self.decision!r})"
