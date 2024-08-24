import argparse
import torch
import datetime
import json
import yaml
import os
import pandas as pd

from main_model import CSDI_Forecasting
from dataset_forecasting import get_dataloader  # Ensure dataloader handles increased dimensionality
from utils import train, evaluate
import yaml

# 코랩에서 GPU를 사용할 수 있도록 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_config():
    with open('/content/drive/Othercomputers/My MacBook Air/Desktop/Dissertation/code/real code/CSDI_final/config/base_forecasting.yaml', 'r') as file:
        return yaml.safe_load(file)
config = load_config()
print(config)


# Argument Parser 설정
parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="/content/drive/Othercomputers/My MacBook Air/Desktop/Dissertation/code/real code/CSDI_final/config/base_forecasting.yaml")
parser.add_argument("--datatype", type=str, default="electricity")
parser.add_argument('--device', default='cuda', help='Device for computation (e.g., cuda)')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--data", type=str, default="/content/drive/Othercomputers/My MacBook Air/Desktop/Dissertation/code/real code/CSDI_final/electricity consumption_daily.csv", help="Path to the CSV file with the data")


args = parser.parse_args()
print(args)

# Config 파일 로드
path = "/content/drive/Othercomputers/My MacBook Air/Desktop/Dissertation/code/real code/CSDI_final/config/base_forecasting.yaml"
with open(path, "r") as f:
    config = yaml.safe_load(f)

# 타겟 차원 설정
if args.datatype == 'electricity':
    target_dim = 320  # 데이터셋의 컬럼 수에 맞게 설정
    config["model"]["is_unconditional"]

#config["model"]["is_unconditional"] = args.unconditional

print(json.dumps(config, indent=4))

# 현재 시간 기반으로 폴더 생성
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/forecasting_" + args.datatype + '_' + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# 데이터 로더 설정
train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    datatype=args.data,
    device=device,
    batch_size=config["train"]["batch_size"],
)

# 모델 초기화 및 디바이스 설정
model = CSDI_Forecasting(config, device, target_dim).to(torch.float32).to(device)

# 모델 학습 또는 불러오기
if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
    # 모델 저장
    torch.save(model.state_dict(), os.path.join("/content/drive/Othercomputers/My MacBook Air/Desktop/Dissertation/code/real code/CSDI_final", "model.pth"))
else:
    model.load_state_dict(torch.load(os.path.join("/content/drive/Othercomputers/My MacBook Air/Desktop/Dissertation/code/real code/CSDI_final", "model.pth")))
model.target_dim = target_dim

# 모델 평가
evaluate(
    model,  # 모델을 첫 번째 인자로 넘김
    test_loader,
    train_loader,  # train_loader는 세 번째 인자로 넘어가야 함
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)
