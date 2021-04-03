import torch
from torch import nn
import torch.nn.functional as F


# class Classifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, use_non_linear):
#         super(Classifier, self).__init__()
#         self.use_non_linear = use_non_linear

#         # Fully connected layer
#         self.lin1 = nn.Linear(input_dim, hidden_dim)
#         self.lin2 = nn.Linear(hidden_dim, hidden_dim)
#         self.lin3 = nn.Linear(hidden_dim, output_dim)
#         #self.lin4 = nn.Linear(int(hidden_dim/2), output_dim)
#         if use_non_linear:
#             self.act1 = nn.ReLU()
#             self.act2 = nn.ReLU()
#             #self.act3 = nn.ReLU()
#         self.log_softmax = torch.nn.LogSoftmax(dim=1)


#     def forward(self, x):
#         x = self.lin1(x)
#         if self.use_non_linear:
#             x = self.act1(x)
#         x = self.lin2(x)
#         if self.use_non_linear:
#             x = self.act2(x)
#         x = self.lin3(x)
#         #if self.use_non_linear:
#         #    x = self.act3(x)
#         #x = self.lin4(x)
#         #x = self.lin3(x)
#         #if self.use_non_linear:
#         #    x = self.act3(x)
#         return self.log_softmax(x)

# def weight_reset(m):
#     if isinstance(m, nn.Linear):
#         m.reset_parameters()

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_non_linear):
        super(Classifier, self).__init__()
        self.use_non_linear = use_non_linear

        # Fully connected layer
        if use_non_linear:
            self.lin1 = nn.Linear(input_dim, hidden_dim)
            self.lin2 = nn.Linear(hidden_dim, hidden_dim)
            self.lin3 = nn.Linear(hidden_dim, output_dim)
            #self.lin4 = nn.Linear(int(hidden_dim/2), output_dim)
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            #self.act3 = nn.ReLU()
        else:
            self.lin1 = nn.Linear(input_dim, output_dim)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.lin1(x)
        if self.use_non_linear:
            x = self.act1(x)
            x = F.dropout(x, p=0.5)
            x = self.lin2(x)
            x = F.dropout(x, p=0.5)
            x = self.act2(x)
            x = self.lin3(x)
        #if self.use_non_linear:
        #    x = self.act3(x)
        #x = self.lin4(x)
        #x = self.lin3(x)
        #if self.use_non_linear:
        #    x = self.act3(x)
        return self.log_softmax(x)

def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()