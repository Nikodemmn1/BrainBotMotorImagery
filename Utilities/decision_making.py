import numpy as np
class DecisionMaker():
    def __init__(self, window_lenght, priorities, thresholds):
        self.priorities = priorities
        self.window_lenght = window_lenght
        self.thresholds = thresholds
        self.data = np.zeros(window_lenght)
        self.decisions = np.zeros(len(thresholds))
    def add_data(self, label):
        self.data = np.roll(self.data, 1)
        self.data[0] = label
    def decide(self):
        ones_mask = np.ones(self.data.shape[0])
        zeros_mask = np.zeros(self.data.shape[0])
        for i in range(len(self.thresholds)):
            mask = np.where(self.data == i, ones_mask, zeros_mask)
            if np.sum(mask) > self.thresholds[i]:
                self.decisions[i] = 1
            else:
                self.decisions[i] = 0
        for label in self.priorities:
            if self.decisions[label] == 1:
                return label




