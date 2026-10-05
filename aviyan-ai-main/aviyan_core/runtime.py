from .identity import IDENTITY
from .memory.store import MemoryStore
from .rag.store import SimpleRAG
from .agents.orchestrator import Orchestrator


class AviyanRuntime:
    def __init__(self):
        self.memory = MemoryStore()
        self.rag = SimpleRAG()
        self.agent = Orchestrator()

    def identity(self):
        return IDENTITY

    def context(self, user_id, text):
        return {
            "memory": self.memory.search(user_id, text),
            "rag": self.rag.search(text),
        }

    def reply(self, user_id, text):
        text_clean = text.strip()
        if text_clean.lower() in {
            "who created you",
            "who is your developer",
            "who is your founder",
            "who is your boss",
        }:
            return (
                f"I was created and developed by {IDENTITY['creator']}, "
                f"Founder & Creator of AVIYAN at {IDENTITY['company']}, from {IDENTITY['origin']}."
            )

        self.memory.add(user_id, text_clean)
        return (
            "AVIYAN is online. I received your request: "
            f"{text_clean}\n\n"
            "The current runtime is a safe foundation build. Connect a trained language model "
            "and generation backends to unlock full production intelligence."
        )
