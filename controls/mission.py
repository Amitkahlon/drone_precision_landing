import numpy as np


class Mission:
    def __init__(self, start, goal):
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
