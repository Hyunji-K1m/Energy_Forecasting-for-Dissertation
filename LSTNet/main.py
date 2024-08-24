import argparse
import math
import time

import torch
import torch.nn as nn
from models import LSTNet
import numpy as np
import importlib

from utils import *
import Optim
from sklearn.metrics import mean_squared_error
def smape(actual, predicted):
    denominator = (torch.abs(actual) + torch.abs(predicted))
    diff = torch.abs(actual - predicted) / denominator
    return 200 * torch.mean(diff)

def mase(actual, predicted):
    n = actual.numel()
    d = torch.mean(torch.abs(actual[1:] - actual[:-1]))  # naive forecast error
    epsilon = 1e-10
    d = d + epsilon  # 0으로 나누는 것을 방지하기 위해 epsilon 추가
    # 예측값과 실제값의 절대 오차
    errors = torch.abs(actual - predicted)
    # MASE 계산
    return torch.mean(errors / d)


def mse(actual, predicted):
    return torch.mean((actual - predicted) ** 2, dim=0)  

def relative_mse(actual, predicted):
    mse = torch.mean((actual - predicted) ** 2)
    actual_mean_squared = torch.mean(actual ** 2)
    return mse / actual_mean_squared


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    n_samples = 0
    total_smape = 0
    total_mase = 0
    total_crps = 0
    total_crps_sum = 0
    total_rmse = 0
    total_relative_rmse = 0
    
    for X, Y in data.get_batches(X, Y, batch_size, False):
        X, Y = X.to(device).float(), Y.to(device).float()
        output = model(X)

        scale = data.scale.expand(output.size(0), data.m).to(device).float()
        scaled_output = output * scale
        scaled_Y = Y * scale

    smape_values = []
    mase_values = []
    mse_values = []
    relative_mse_values = []
    num_features = scaled_output.size(1)

    for i in range(scaled_output.size(1)):  # 피처 수만큼 반복
        feature_output = scaled_output[:, i]  # i번째 피처의 예측값
        feature_Y = scaled_Y[:, i]  # i번째 피처의 실제값

        smape_values.append(smape(feature_Y, feature_output).item())
        mase_values.append(mase(feature_Y, feature_output).item())
        mse_values.append(mse(feature_Y, feature_output).item())  # MSE 값을 추가
        relative_mse_values.append(relative_mse(feature_Y, feature_output).item())

    smape_value = torch.tensor(smape_values).mean().item()
    mase_value = torch.tensor(mase_values).mean().item()
    rmse_values = torch.sqrt(torch.tensor(mse_values))  # 각 피처별 RMSE 계산
    relative_rmse_values = torch.sqrt(torch.tensor(relative_mse_values))  # 각 피처별 Relative RMSE 계산

    rmse_value = rmse_values.mean().item()  # 피처별 RMSE 평균
    relative_rmse_value = relative_rmse_values.mean().item()  # 피처별 Relative RMSE 평균


    return smape_value, mase_value, rmse_value, relative_rmse_value, scaled_output, scaled_Y


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        X, Y = X.to(device).float(), Y.to(device).float()  
        model.zero_grad()
        output = model(X)
        scale = data.scale.expand(output.size(0), data.m).to(device).float()  
        loss = criterion(output * scale, Y * scale)  # 원본 데이터로 loss 계산
        loss.backward()
        grad_norm = optim.step()
        total_loss += loss.item()
        n_samples += (output.size(0) * data.m)
    return total_loss / n_samples


    
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument("--data", type=str, default="/Users/kimhyunji/Desktop/Dissertation/code/data/rea_final_data without seasonal.csv", help="Path to the CSV file with the data")
parser.add_argument('--model', type=str, default='LSTNet',
                    help='')
parser.add_argument('--hidCNN', type=int, default=10,
                    help='number of CNN hidden units')
parser.add_argument('--hidRNN', type=int, default=10,
                    help='number of RNN hidden units')
parser.add_argument('--window', type=int, default=365,
                    help='window size') 
parser.add_argument('--CNN_kernel', type=int, default=6,
                    help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=365,
                    help='The window size of the highway component')
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='/Users/kimhyunji/Desktop/Dissertation/code/real code/LSTNet Why english/model.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--horizon', type=int, default=7) #predict_length=30 or 14 7
parser.add_argument('--skip', type=float, default=120)#365 
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')
args = parser.parse_args()

# 디바이스 선택 (MPS가 가능한지 확인 후 설정)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
    
Data = Data_utility(args.data, 0.7, 0.15, args.cuda, args.horizon, args.window, args.normalize)

model = eval(args.model).Model(args, Data).to(device)

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

if args.L1Loss:
    criterion = nn.L1Loss(reduction='sum')
else:
    criterion = nn.MSELoss(reduction='sum')

evaluateL2 = nn.MSELoss(reduction='sum')
evaluateL1 = nn.L1Loss(reduction='sum')

# Move criterion and evaluation to the right device
criterion = criterion.to(device)
evaluateL1 = evaluateL1.to(device)
evaluateL2 = evaluateL2.to(device)

best_val = float('inf')
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip,
)

try:
    print('begin training')
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # 학습
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)

        # 검증 - 원본 스케일로 변환된 값을 사용
        val_smape, val_mase, val_rmse, val_rel_rmse, val_predict, val_test = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size)

        # 성능 평가 지표 출력
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid sMAPE {:5.4f} | valid MASE {:5.4f} | valid RMSE {:5.4f} | valid Relative RMSE {:5.4f}'.format(
            epoch, (time.time() - epoch_start_time), train_loss, val_smape, val_mase,val_rmse, val_rel_rmse))

        # 성능이 개선되었을 때 모델 저장
        if val_smape < best_val:
            with open(args.save, 'wb') as f:
                torch.save(model.state_dict(), f)
            best_val = val_smape

        # 매 5 에포크마다 테스트 데이터로 성능 평가
        if epoch % 5 == 0:
            test_smape, test_mase, test_rmse, test_rel_rmse, test_predict, test_test = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
            print("test sMAPE {:5.4f} | test MASE {:5.4f} | test RMSE {:5.4f} | test Relative RMSE {:5.4f}".format(
                test_smape, test_mase, test_rmse, test_rel_rmse))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# 학습 완료 후 모델 불러오기
with open(args.save, 'rb') as f:
    model.load_state_dict(torch.load(f))
model.to(device)

# 테스트 데이터에 대한 최종 성능 평가
test_smape, test_mase, test_rmse, test_rel_rmse, test_predict, test_test = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
print("test sMAPE {:5.4f} | test MASE {:5.4f} | test RMSE {:5.4f} | test Relative RMSE {:5.4f}".format(
    test_smape, test_mase, test_rmse, test_rel_rmse))
