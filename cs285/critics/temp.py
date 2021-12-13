import torch

terminal = [0,0,0,1,0,0,1,0,0,0,1,0,1]
terminal = torch.tensor(terminal)
states = [1,2,3,4,5,76,7,8,8,9,1,2,3]
states = torch.tensor(states)
ret = [4,3,4,2]

idx = torch.nonzero(terminal).squeeze() + 1
print(idx)
print(torch.tensor_split(states, idx))