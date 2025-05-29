import torch

data = torch.load("model_output_test.pt")


print(data[0].type())
