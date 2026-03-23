import json
import os


class Logger:
    def __init__(self, save_path="results/log.json"):
        self.save_path = save_path
        self.data = {
            "episode": [],
            "distance": [],
            "success": [],
            "success_rate": [],
            "difficulty": []
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def log(self, episode, distance, success, success_rate, difficulty):
        self.data["episode"].append(episode)
        self.data["distance"].append(distance)
        self.data["success"].append(int(success))
        self.data["success_rate"].append(success_rate)
        self.data["difficulty"].append(difficulty)

    def save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.data, f, indent=4)

        print(f"📁 Logs saved to {self.save_path}")