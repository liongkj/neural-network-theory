from torch import nn,optim,functional as F
import torch

class QuadraticLayer(nn.Module):
    def __init__(self,in_size,out_size,logging):
        super().__init__()
        self.logging = logging
        self.in_size, self.out_size = in_size,out_size
        self.weights_u = nn.Parameter(torch.zeros(in_size, out_size))
        self.weights_v = nn.Parameter(torch.zeros(in_size, out_size))
        self.bias = nn.Parameter(torch.zeros(out_size))
        init_var = 20
        init_mean = -15
#         nn.init.uniform_(self.weights_u,-init_val,init_val)
#         nn.init.uniform_(self.weights_v,-init_val,init_val)
        nn.init.normal_(self.weights_u,init_mean,init_var)
        nn.init.normal_(self.weights_v,init_mean,init_var)
           
    def forward(self, x):
        out = torch.mm(x,self.weights_v) + torch.mm(x**2,self.weights_u) + self.bias
        if(self.logging):
            print('x.shape:',x.shape)
            print('weights:',self.weights_u)
            print('bias:',self.bias.shape)
            print('output shape:',out.shape)
        return out
    
    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.in_size, self.out_size, self.bias is not None
        )

class Mlqp(nn.Module):
    def __init__(self, num_features,num_hidden,num_output,logging=False):
        super().__init__()
        self.layers = nn.Sequential(
            QuadraticLayer(num_features,num_hidden,logging),
            nn.Sigmoid(),
            nn.Linear(num_hidden,num_output, bias=False)
        ) 
    def forward(self,x):
        output = self.layers(x)
        return output;