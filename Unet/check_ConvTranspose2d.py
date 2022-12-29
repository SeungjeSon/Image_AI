### 출저 : https://cumulu-s.tistory.com/29 ###

import torch
import torch.nn as nn
import numpy as np

# sample value
test_input = torch.Tensor([[[[1,2,3],[4,5,6]]]])
# print("input size", test_input.shape)
# print("test input", test_input)

class sample(nn.Module):
    def __init__(self):
        super(sample, self).__init__()
        self.main = nn.ConvTranspose2d(1,3,4,1,0, bias=False) # (1,3,4,1,0) = (ch of input, ch of output, kernel size, stride, padding)

    def forward(self, input):
        return self.main(input)

Model = sample()

# Print model`s original parameters
# for name, params in Model.state_dict().items():
    # print("name : {0}\nParams : {1}\nParams shape : {2}".format(name, params, params.shape))

# I makes 48 values from 0.1 to 4.8 and make (1, 3, 4, 4) shape
np_sam = np.linspace(0.1, 4.8, num = 48)
np_sam_torch = torch.Tensor(np_sam)
sam_tor = np_sam_torch.view(1, 3, 4, 4)

# Modify model`s params using 4 for loops
with torch.no_grad():
    batch, channel, width, height = Model.main.weight.shape

    for b in range(batch):
        for c in range(channel):
            for w in range(width):
                for h in range(height):
                    Model.main.weight[b][c][w][h] = sam_tor[b][c][w][h]

# Check parameter modification.
# print("Model weight: ", Model.main.weight)

result = Model(test_input)

print("Result shape: ", result.shape)
print("Result: ", result)