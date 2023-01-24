import numpy as np
class DecisionMaker():
    def __init__(self, window_length, priorities, thresholds):
        self.priorities = priorities
        self.window_length = window_length
        self.thresholds = thresholds
        self.data = np.zeros(window_length)
        self.decisions = np.zeros(len(thresholds))
        self.decisions_masks = np.zeros(len(thresholds))
        self.prev_decisions_masks = np.zeros(len(thresholds))
    def add_data(self, label):
        self.data = np.roll(self.data, 1)
        self.data[0] = label
    def decide(self):
        ones_mask = np.ones(self.data.shape[0])
        zeros_mask = np.zeros(self.data.shape[0])
        for i in range(len(self.thresholds)):
            mask = np.where(self.data == i, ones_mask, zeros_mask)
            self.decisions_masks[i] = np.sum(mask) / self.window_length
            if np.sum(mask) > self.thresholds[i] * self.window_length:
                self.decisions[i] = 1
            else:
                self.decisions[i] = 0
            self.prev_decisions_masks[i] = self.decisions_masks[i]
        for label in np.flip(np.argsort(self.decisions_masks)):
            if self.decisions[label] == 1:
                return label
        return None

class BinaryDecisionMaker():
    def __init__(self, priorities, thresholds):
        self.priorities = priorities
        self.threshols = thresholds

    def decide(self, decisions):
        mask = decisions > self.threshols
        for i in mask

