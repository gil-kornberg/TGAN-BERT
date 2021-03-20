import torch
from torch import nn

seqLen = 50
embedSize = 768
batchSize = 10
outputSize = seqLen * embedSize
hidden_size = 300  # dimension of hidden layer size
noiseDim = 1024  # dimension of noise vector
device = 'cuda'

def make_gen_block(inputDim=noiseDim, outputDim=hidden_size):
    return nn.Sequential(
        nn.Linear(in_features=inputDim, out_features=outputDim),
        nn.LayerNorm(outputDim),
        nn.LeakyReLU(negative_slope=0.2, inplace=False),
        nn.Dropout(p=0.5, inplace=False)
    )


class Generator(nn.Module):

    def __init__(self, inputDim=noiseDim,
                 outputDim=hidden_size):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            make_gen_block(inputDim, outputDim),
            make_gen_block(outputDim, outputDim),
            make_gen_block(outputDim, outputDim),
            nn.Linear(in_features=hidden_size, out_features=outputSize),  # this last outputdim is max_len*embed_size
        )

    # given a noise tensor return a generated "BERT" embedding
    def forward(self, noise):
        return self.gen(noise).to(device)


def get_noise(z_dim):
    t = torch.empty(z_dim)
    nn.init.uniform_(t, a=0.0, b=1.0)
    return t.to(device)
    #return torch.randn(z_dim, device=device)
