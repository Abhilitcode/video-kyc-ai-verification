import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Active GPU:", torch.cuda.get_device_name(0))
