# create a role Enum which container two variable: AI, Human
from enum import Enum

class Role(Enum):
    AI = "AI"
    Human = "Human"


def serialize_history(history: list) -> str:
    """Serialize a conversation history to a string."""
    serialized_history = ""
    for role, message in history:
        serialized_history += f"{role.value}: {message}\n"
    return serialized_history

def deserialize_history(serialized_history: str) -> list:
    """Deserialize a conversation history from a string."""
    history = []
    for line in serialized_history.split("\n"):
        if line:
            role, message = line.split(": ")
            history.append((Role(role), message))
    return history


def add_msg_to_serialized_history(history: str, role: Role, message: str) -> str:
    """Add a message to a serialized conversation history."""
    return history + f"{role.value}: {message}\n"

def add_msg_to_deserialized_history(history: list, role: Role, message: str) -> list:
    """Add a message to a list conversation history."""
    history.append((role, message))
    return history