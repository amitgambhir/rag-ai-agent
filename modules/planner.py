from typing import List, Dict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv

class Planner:
    """
    Simple Planner to manage multi-step workflows or tasks.
    Designed for extensibility.
    """

    def __init__(self):
        self.task_queue: List[Dict] = []
        self.completed_tasks: List[Dict] = []

    def plan (self, question: str):
        # Dummy plan logic - you can improve it later
        return [
            f"Step 1: Understand the question '{question}'.",
            "Step 2: Search relevant documents.",
            "Step 3: Summarize findings.",
            "Step 4: Provide final answer."
        ]

    def add_task(self, task_name: str, params: Dict = None):
        """
        Add a new task to the queue.
        """
        if params is None:
            params = {}
        task = {"name": task_name, "params": params, "status": "pending"}
        self.task_queue.append(task)

    def get_next_task(self):
        """
        Retrieve the next pending task.
        """
        for task in self.task_queue:
            if task["status"] == "pending":
                return task
        return None

    def mark_task_completed(self, task_name: str):
        """
        Mark task as completed and move to completed list.
        """
        for task in self.task_queue:
            if task["name"] == task_name and task["status"] == "pending":
                task["status"] = "completed"
                self.completed_tasks.append(task)
                self.task_queue.remove(task)
                return True
        return False

    def reset(self):
        """
        Reset planner state.
        """
        self.task_queue.clear()
        self.completed_tasks.clear()

    def get_status(self):
        """
        Get current planner status.
        """
        return {
            "pending_tasks": [t for t in self.task_queue if t["status"] == "pending"],
            "completed_tasks": self.completed_tasks,
        }
    
