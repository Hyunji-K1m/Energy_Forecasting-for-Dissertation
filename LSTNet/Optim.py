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
    n = actual.size(0)
    d = torch.mean(torch.abs(actual[1:] - actual[:-1]))  
    #epsilon = 1e-10
    d = d #+ epsilon 
    errors = torch.abs(actual - predicted)
    return torch.mean(errors / d)

def crps(actual, predicted):
    return torch.mean((predicted - actual) ** 2)

def crps_sum(actual, predicted):
    return torch.sum((predicted - actual) ** 2)

def rmse(actual, predicted):
    return torch.sqrt(torch.mean((actual - predicted) ** 2))

def relative_rmse(actual, predicted):
    rmse_val = rmse(actual, predicted)
    mean_actual = torch.mean(actual)
    return rmse_val / mean_actual

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    
    for X, Y in data.get_batches(X, Y, batch_size, False):
        X, Y = X.to(device).float(), Y.to(device).float()  
        output = model(X)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m).to(device).float()
        scaled_output = output * scale  
        scaled_Y = Y * scale  

        total_loss += evaluateL2(scaled_output, scaled_Y).item()
        total_loss_l1 += evaluateL1(scaled_output, scaled_Y).item()
        n_samples += (output.size(0) * data.m)

    smape_value = smape(scaled_Y, scaled_output)
    mase_value = mase(scaled_Y, scaled_output)
    crps_value = crps(scaled_Y, scaled_output)
    crps_sum_value = crps_sum(scaled_Y, scaled_output)
    rmse_value = rmse(scaled_Y, scaled_output)
    relative_rmse_value = relative_rmse(scaled_Y, scaled_output)

    return smape_value, mase_value, crps_value, crps_sum_value, rmse_value, relative_rmse_value, scaled_output, scaled_Y


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
parser.add_argument("--data", type=str, default="DATA.csv", help="Path to the CSV file with the data")
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
parser.add_argument('--epochs', type=int, default=50,
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
parser.add_argument('--save', type=str,  default='FILE.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--horizon', type=int, default=14) #predict_length=30 or 14 7
parser.add_argument('--skip', type=float, default=120)#365 
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')
args = parser.parse_args()

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

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

        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)

        val_smape, val_mase, val_crps, val_crps_sum, val_rmse, val_rel_rmse, val_predict, val_test = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size)

        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid sMAPE {:5.4f} | valid MASE {:5.4f} | valid CRPS {:5.4f} | valid CRPS_sum {:5.4f} | valid RMSE {:5.4f} | valid Relative RMSE {:5.4f}'.format(
            epoch, (time.time() - epoch_start_time), train_loss, val_smape, val_mase, val_crps, val_crps_sum, val_rmse, val_rel_rmse))

        if val_smape < best_val:
            with open(args.save, 'wb') as f:
                torch.save(model.state_dict(), f)
            best_val = val_smape

        if epoch % 5 == 0:
            test_smape, test_mase, test_crps, test_crps_sum, test_rmse, test_rel_rmse, test_predict, test_test = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
            print("test sMAPE {:5.4f} | test MASE {:5.4f} | test CRPS {:5.4f} | test CRPS_sum {:5.4f} | test RMSE {:5.4f} | test Relative RMSE {:5.4f}".format(
                test_smape, test_mase, test_crps, test_crps_sum, test_rmse, test_rel_rmse))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

with open(args.save, 'rb') as f:
    model.load_state_dict(torch.load(f))
model.to(device)

test_smape, test_mase, test_crps, test_crps_sum, test_rmse, test_rel_rmse, test_predict, test_test = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
print("test sMAPE {:5.4f} | test MASE {:5.4f} | test CRPS {:5.4f} | test CRPS_sum {:5.4f} | test RMSE {:5.4f} | test Relative RMSE {:5.4f}".format(
    test_smape, test_mase, test_crps, test_crps_sum, test_rmse, test_rel_rmse))
