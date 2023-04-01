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


class BinaryDecisionMaker:
    def __init__(self, window_length, thresholds, noise_stop):
        self.window_length = window_length
        self.thresholds = thresholds
        self.data = np.zeros((window_length, 4))
        self.decisions = np.zeros(len(thresholds))
        self.average_decision = np.zeros(len(thresholds))
        self.noise_stop = noise_stop

    def add_data(self, decision):
        self.data = np.roll(self.data, 1, axis=0)
        self.data[0, :] = decision

    def decide(self):
        self.average_decision = self.data.mean(axis=0)
        if self.noise_stop and self.average_decision[3] >= self.thresholds[3]:
            return None
        over_thresholds = self.average_decision - self.thresholds
        if (over_thresholds >= 0.0).any():
            return np.argmax(over_thresholds)
        return None


