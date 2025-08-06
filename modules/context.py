# context manager
# Placeholder for shared execution context between modules

class AgentContext:
    """
    Global context object to maintain shared state across the AI Agent's lifecycle.
    """

    def __init__(self):
        self.chat_history = []       # Tracks conversation history
        self.documents = []          # Loaded documents (if applicable)
        self.current_task = None     # Task currently being executed
        self.user_profile = {}       # User-specific metadata (e.g., preferences, plan type)

    def add_chat(self, role, message):
        self.chat_history.append({"role": role, "message": message})

    def get_chat_history(self):
        return self.chat_history

    def set_user_profile(self, profile_dict):
        self.user_profile.update(profile_dict)

    def get_user_profile(self):
        return self.user_profile

    def set_task(self, task):
        self.current_task = task

    def get_task(self):
        return self.current_task

    def reset(self):
        self.chat_history.clear()
        self.documents.clear()
        self.current_task = None
        self.user_profile.clear()
