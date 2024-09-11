import argparse
import torch
import datetime
import json
import yaml
import os
import pandas as pd

from main_model import CSDI_Forecasting
from dataset_forecasting import get_dataloader  
from utils import train, evaluate
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_config():
    with open('FILE.yaml', 'r') as file:
        return yaml.safe_load(file)
config = load_config()
print(config)


parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="FILE.yaml")
parser.add_argument("--datatype", type=str, default="electricity")
parser.add_argument('--device', default='cuda', help='Device for computation (e.g., cuda)')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--data", type=str, default="DATA.csv", help="Path to the CSV file with the data")


args = parser.parse_args()
print(args)

path = "FILE.yaml"
with open(path, "r") as f:
    config = yaml.safe_load(f)

if args.datatype == 'electricity':
    target_dim = 320  
    config["model"]["is_unconditional"]

#config["model"]["is_unconditional"] = args.unconditional

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/forecasting_" + args.datatype + '_' + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    datatype=args.data,
    device=device,
    batch_size=config["train"]["batch_size"],
)

model = CSDI_Forecasting(config, device, target_dim).to(torch.float32).to(device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
    torch.save(model.state_dict(), os.path.join("PATH", "model.pth"))
else:
    model.load_state_dict(torch.load(os.path.join("PATH", "model.pth")))
model.target_dim = target_dim


evaluate(
    model,  
    test_loader,
    train_loader, 
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)
