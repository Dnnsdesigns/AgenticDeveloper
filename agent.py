import numpy as np

class SecondaryAgent:
    def __init__(self, model, specialty):
        self.model = model
        self.specialty = specialty
    
    def predict(self, state):
        return self.model.predict(state)

class PrimeAgent:
    def __init__(self, gating_network, experts):
        self.gating_network = gating_network
        self.experts = experts
    
    def act(self, state):
        gating_weights = self.gating_network.predict(state)
        expert_outputs = [expert.predict(state) for expert in self.experts]
        
        # Weighted sum of expert outputs based on gating weights
        combined_output = np.sum([weight * output for weight, output in zip(gating_weights[0], expert_outputs)], axis=0)
        action = np.argmax(combined_output)
        return action
    
    def train(self, states, actions, rewards):
        self.gating_network.fit(states, actions, sample_weight=rewards, epochs=1, verbose=0)
        for expert in self.experts:
            expert.model.fit(states, actions, sample_weight=rewards, epochs=1, verbose=0)
