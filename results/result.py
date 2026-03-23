import json
import matplotlib.pyplot as plt


def plot_results(log_path="results/log.json"):

    with open(log_path, "r") as f:
        data = json.load(f)

    episodes = data["episode"]
    distance = data["distance"]
    success_rate = data["success_rate"]
    difficulty = data["difficulty"]

    # -------- Plot 1: Distance --------
    plt.figure()
    plt.plot(episodes, distance)
    plt.title("Distance to Target")
    plt.xlabel("Episode")
    plt.ylabel("Distance")
    plt.grid()
    plt.show()

    # -------- Plot 2: Success Rate --------
    plt.figure()
    plt.plot(episodes, success_rate)
    plt.title("Success Rate (last 20)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.grid()
    plt.show()

    # -------- Plot 3: Difficulty --------
    plt.figure()
    plt.plot(episodes, difficulty)
    plt.title("Curriculum Difficulty")
    plt.xlabel("Episode")
    plt.ylabel("Difficulty Level")
    plt.yticks([0, 1, 2], ["Easy", "Medium", "Hard"])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_results()