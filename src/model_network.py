import torch.nn as nn

class DailyEnergyForecaster(nn.Module):
    """Feedforward Neural Network for Energy Forecasting."""
    def __init__(self, input_dim):
        super(DailyEnergyForecaster, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)