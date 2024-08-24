import os
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import os
import yaml

# 코랩에서 GPU를 사용할 수 있도록 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_config():
    with open('/content/drive/Othercomputers/My MacBook Air/Desktop/Dissertation/code/real code/CSDI_final/config/base_forecasting.yaml', 'r') as file:
        return yaml.safe_load(file)
config = load_config()

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="/content/drive/Othercomputers/My MacBook Air/Desktop/Dissertation/code/real code/CSDI_final",
):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "/content/drive/Othercomputers/My MacBook Air/Desktop/Dissertation/code/real code/CSDI_final":
        output_path = foldername + "/model.pth"
    else:
        output_path = os.path.join("/content/drive/Othercomputers/My MacBook Air/Desktop/Dissertation/code/real code/CSDI_final", "model.pth")

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                train_batch = {key: value.to(device, dtype=torch.float32) for key, value in train_batch.items()}

                optimizer.zero_grad()

                loss = model(train_batch)
                print("Loss value:", loss.item())
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        valid_batch = {key: value.to(device, dtype=torch.float32) for key, value in valid_batch.items()}
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)

def calc_denominator(target, eval_points):
    denom = torch.sum(torch.abs(target * eval_points))
    if denom == 0:
        denom += 1e-8
    return torch.sum(torch.abs(target * eval_points))

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

# sMAPE 계산 함수
def smape(target, forecast, eval_points):
    numerator = torch.abs(forecast - target)
    denominator = torch.abs(forecast) + torch.abs(target)
    smape_val = 200 * torch.sum(numerator * eval_points / denominator) / torch.sum(eval_points)
    return smape_val

# MASE 계산 함수
def mase(target, forecast, eval_points, training_series):
    numerator = torch.sum(torch.abs((forecast - target) * eval_points))
    n = training_series.size(0)
    d = torch.sum(torch.abs(training_series[1:] - training_series[:-1]))
    mase_val = numerator / (d / (n - 1))
    return mase_val

def evaluate(model, test_loader, train_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    model.to(device)
    with torch.no_grad():
        model.eval()
        smape_total = 0
        mase_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        training_series = []
        for batch_no, train_batch in enumerate(train_loader, start=1):
            print(train_batch.keys())
            for key, value in train_batch.items():
                print(f"Key: {key}, Shape: {value.shape}")  # 각 키와 해당 값의 shape를 출력합니다.
                break 
            train_batch = {k: v.to(device, dtype=torch.float32) for k, v in train_batch.items()}
            training_series.append(train_batch['observed_data'])
        training_series = torch.cat(training_series, dim=0)

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                test_batch ={k: v.to(device, dtype=torch.float32) for k, v in test_batch.items()}
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                smape_current = smape(c_target, samples_median.values, eval_points)
                mase_current = mase(c_target, samples_median.values, eval_points, training_series.sum(-1))

                smape_total += smape_current.item()
                mase_total += mase_current.item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "sMAPE_total": smape_total / batch_no,
                        "MASE_total": mase_total / batch_no,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        smape_total / evalpoints_total,
                        mase_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("sMAPE:", smape_total / evalpoints_total)
                print("MASE:", mase_total / evalpoints_total)
                print("CRPS:", CRPS)
                print("CRPS_sum:", CRPS_sum)
