import torch
from torch import nn

class NonLinearModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearModel, self).__init__()
        
        # Fully connected layer
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.Sigmoid()
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.act2 = nn.Sigmoid()
        self.log_softmax = torch.nn.LogSoftmax() #correct dimension? dim =1 gives negative NLL

    def forward(self, x):
        return self.log_softmax(self.act2(self.lin2(self.act1(self.lin1(x)))))

def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()
        
