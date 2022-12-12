import torch

checkpoint_path = "models/lightning_logs/version_577472/epoch=0-step=3303.ckpt"

checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
print(checkpoint["hyper_parameters"])
print("epoch: ", checkpoint["epoch"])
print(checkpoint.keys())

torch.save(checkpoint["state_dict"], "models/finetune.pt") 
