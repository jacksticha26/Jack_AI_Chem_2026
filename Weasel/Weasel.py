import random
import string
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Target phrase
TARGET = "Methinks it is like a weasel"

# Mutation characters (A-Z, a-z, space)
MUTATION_CHARS = string.ascii_letters + " "


def fitness(candidate: str) -> int:
    return sum(1 for c, t in zip(candidate, TARGET) if c == t)


def mutate(candidate: str, mutation_rate: float = 0.05) -> str:
    return "".join(
        random.choice(MUTATION_CHARS) if random.random() < mutation_rate else ch
        for ch in candidate
    )


def random_string(length: int) -> str:
    return "".join(random.choice(MUTATION_CHARS) for _ in range(length))


class EvolutionApp:
    def __init__(self, root):
        self.root = root
        root.title("Evolution Simulator")

        # Evolution state
        self.parent = random_string(len(TARGET))
        self.best_score = fitness(self.parent)
        self.generation = 0
        self.fitness_history = []

        # UI layout
        self.setup_ui()

        # Initial plot
        self.update_plot()

    def setup_ui(self):
        # Frame for controls
        control = ttk.Frame(self.root)
        control.pack(pady=10)

        # Entry box for manual fitness check
        self.entry = ttk.Entry(control, width=40)
        self.entry.pack(side=tk.LEFT)
        self.entry.insert(0, "Enter phrase here")

        self.check_button = ttk.Button(control, text="Score", command=self.score_entry)
        self.check_button.pack(side=tk.LEFT, padx=5)

        self.result_label = ttk.Label(control, text="Score: ")
        self.result_label.pack(side=tk.LEFT)

        # Evolution button
        self.evolve_button = ttk.Button(self.root, text="Next Generation", command=self.evolve_one)
        self.evolve_button.pack(pady=5)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

    def score_entry(self):
        text = self.entry.get()
        score = fitness(text)
        self.result_label.config(text=f"Score: {score}")

    def evolve_one(self):
        # Create offspring
        offspring = [mutate(self.parent) for _ in range(100)]

        # Score them
        scored = [(fitness(child), child) for child in offspring]
        score, best = max(scored, key=lambda x: x[0])

        # Replace parent if better
        if score > self.best_score:
            self.parent = best
            self.best_score = score

        # Update history
        self.generation += 1
        self.fitness_history.append(self.best_score)

        # Update UI
        self.update_plot()
        print(f"Gen {self.generation} | Score {self.best_score} | {self.parent}")

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.fitness_history)
        self.ax.set_title("Fitness Over Time")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = EvolutionApp(root)
    root.mainloop()