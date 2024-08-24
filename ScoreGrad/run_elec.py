import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from configs.config import get_configs
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from score_sde.score_sde_estimator import ScoreGradEstimator
from gluonts.dataset.common import ListDataset
# from pts import Trainer
from score_sde.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator
from utils import *
from score_sde.util import seed_torch
from sklearn.preprocessing import MinMaxScaler
from gluonts.model.forecast import Forecast
from gluonts.model.forecast import SampleForecast
"""
def plot(target, forecast, prediction_length, prediction_intervals=(50.0, 90.0), color='g', fname=None):
    label_prefix = ""
    rows = 4
    cols = 4
    fig, axs = plt.subplots(rows, cols, figsize=(24, 24))
    axx = axs.ravel()
    seq_len, target_dim = target.shape

    ps = [50.0] + [
        50.0 + f * c / 2.0 for c in prediction_intervals for f in [-1.0, +1.0]
    ]

    percentiles_sorted = sorted(set(ps))

    def alpha_for_percentile(p):
        return (p / 100.0) ** 0.3

    for dim in range(0, min(rows * cols, target_dim)):
        ax = axx[dim]

        target[-2 * prediction_length:][dim].plot(ax=ax)

        ps_data = [forecast.quantile(p / 100.0)[:, dim] for p in percentiles_sorted]
        i_p50 = len(percentiles_sorted) // 2

        p50_data = ps_data[i_p50]
        p50_series = pd.Series(data=p50_data, index=forecast.index)
        p50_series.plot(color=color, ls="-", label=f"{label_prefix}median", ax=ax)

        for i in range(len(percentiles_sorted) // 2):
            ptile = percentiles_sorted[i]
            alpha = alpha_for_percentile(ptile)
            ax.fill_between(
                forecast.index,
                ps_data[i],
                ps_data[-i - 1],
                facecolor=color,
                alpha=alpha,
                interpolate=True,
            )
            # Hack to create labels for the error intervals.
            # Doesn't actually plot anything, because we only pass a single data point
            pd.Series(data=p50_data[:1], index=forecast.index[:1]).plot(
                color=color,
                alpha=alpha,
                linewidth=10,
                label=f"{label_prefix}{100 - ptile * 2}%",
                ax=ax,
            )

    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    axx[0].legend(legend, loc="upper left")

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.05)
"""

def inverse_transform_forecasts(forecasts, scaler):
    inverse_forecasts = []
    for forecast in forecasts:
        if isinstance(forecast, SampleForecast): 
            forecast_mean = forecast.mean  
            forecast_mean_inverse = scaler.inverse_transform(forecast_mean)  
            inverse_forecasts.append(forecast_mean_inverse)
        elif isinstance(forecast, np.ndarray):
            forecast_mean = np.mean(forecast, axis=0)  
            forecast_mean_inverse = scaler.inverse_transform(forecast_mean)
            inverse_forecasts.append(forecast_mean_inverse)
        else:
            raise ValueError("Unsupported forecast type")
    return inverse_forecasts

def inverse_transform_targets(targets, scaler):
    inverse_targets = []
    for target in targets:
        if isinstance(target, pd.DataFrame): 
            target_values = target.values 
            target_inverse = scaler.inverse_transform(target_values) 
            inverse_targets.append(target_inverse)
        elif isinstance(target, np.ndarray):
            target_inverse = scaler.inverse_transform(target) 
            inverse_targets.append(target_inverse)
        else:
            raise ValueError("Unsupported target type")
    return inverse_targets




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Available datasets: {list(dataset_recipes.keys())}")

file_path = 'data without seasonal.csv'
dataset = pd.read_csv(file_path)
dataset['start_date'] = pd.to_datetime(dataset['start_date'])
dataset.set_index('start_date', inplace=True)

#scaler = MinMaxScaler()
#scaled_data = scaler.fit_transform(dataset)
#dataset = pd.DataFrame(scaled_data, columns=dataset.columns, index=dataset.index)

train_size = 0.7
valid_size = 0.15
test_size = 0.15

train_data, temp_data = train_test_split(dataset, test_size=(1 - train_size), shuffle=False)
valid_data, test_data = train_test_split(temp_data, test_size=(test_size / (valid_size + test_size)), shuffle=False)
train_index = train_data.index
valid_index = valid_data.index
test_index = test_data.index

scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
valid_data = scaler.transform(valid_data)
test_data = scaler.transform(test_data)
train_data = pd.DataFrame(train_data, columns=dataset.columns, index=train_index)
valid_data = pd.DataFrame(valid_data, columns=dataset.columns, index=valid_index)
test_data = pd.DataFrame(test_data, columns=dataset.columns, index=test_index)

train_dataset = ListDataset(
    [{"start": train_data.index[0], "target": train_data[col].values} for col in train_data.columns],
    freq="1D"
)

valid_dataset = ListDataset(
    [{"start": valid_data.index[0], "target": valid_data[col].values} for col in valid_data.columns],
    freq="1D"
)

test_dataset = ListDataset(
    [{"start": test_data.index[0], "target": test_data[col].values} for col in test_data.columns],
    freq="1D"
)

train_grouper = MultivariateGrouper(max_target_dim=train_data.shape[1])
valid_grouper = MultivariateGrouper(max_target_dim=valid_data.shape[1])
test_grouper = MultivariateGrouper(max_target_dim=test_data.shape[1])

args = parse_args()
config = get_configs(dataset=args.data, name=args.name)
seed_torch(config.training.seed)


dataset_train = train_grouper(train_dataset)
dataset_valid = valid_grouper(valid_dataset)
dataset_test = test_grouper(test_dataset)


estimator = ScoreGradEstimator(
    input_size=config.input_size,
    freq='1D',
    prediction_length=30,#30
    target_dim=320,
    context_length=365,
    num_layers=config.num_layers,
    num_cells=config.num_cells,
    cell_type='GRU',
    num_parallel_samples=config.num_parallel_samples,
    dropout_rate=config.dropout_rate,
    conditioning_length=config.conditioning_length,
    diff_steps=config.modeling.num_scales,
    beta_min=config.modeling.beta_min,
    beta_end=config.modeling.beta_max,
    residual_layers=config.modeling.residual_layers,
    residual_channels=config.modeling.residual_channels,
    dilation_cycle_length=config.modeling.dilation_cycle_length,
    #scaling=config.modeling.scaling,
    md_type=config.modeling.md_type,
    continuous=config.training.continuous,
    reduce_mean=config.reduce_mean,
    likelihood_weighting=config.likelihood_weighting,
    config=config,
    trainer=Trainer(
        epochs=config.epochs,
        batch_size=config.batch_size,
        num_batches_per_epoch=config.num_batches_per_epoch,
        learning_rate=config.learning_rate,
        decay=config.weight_decay,
        device=config.device,
        wandb_mode='disabled',
        config=config)

)

if config.train:
    train_output = estimator.train_model(dataset_train, num_workers=0)
    predictor = train_output.predictor


    forecast_it_valid, ts_it_valid = make_evaluation_predictions(
        dataset=dataset_valid,
        predictor=predictor,
        num_samples=100
    )
    
    forecasts_valid = list(forecast_it_valid)
    targets_valid = list(ts_it_valid)
    evaluator_valid = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],target_agg_funcs={'sum': np.sum}
    )
    
    agg_metric_valid, item_metrics_valid = evaluator_valid(targets_valid, forecasts_valid, num_series=len(dataset_valid))

    print(f"First forecast type: {type(forecasts_valid[0])}")
    print(f"First target type: {type(targets_valid[0])}")

    forecasts_valid = inverse_transform_forecasts(forecasts_valid, scaler)
    targets_valid = inverse_transform_targets(targets_valid, scaler)
    
  
    mae_list_valid = []
    mase_list_valid = []
    mse_list_valid = []
    rel_rmse_list_valid = []

    for target, forecast in zip(targets_valid, forecasts_valid):
        forecast_len = len(forecast)
        target = target[:forecast_len]

        mae = np.mean(np.abs(forecast - target))
        mae_list_valid.append(mae)

        naive_mae = np.mean(np.abs(np.diff(target)))
        epsilon = 1e-10  
        mase = mae / (naive_mae + epsilon)
        mase_list_valid.append(mase)
        #mse -> after RMSE 
        mse = np.mean((forecast - target) ** 2)
        mse_list_valid.append(mse)

        mean_target = np.mean(target)
        rel_rmse = np.sqrt(mse) / (mean_target + epsilon)
        rel_rmse_list_valid.append(rel_rmse)

    agg_mase_valid = np.mean(mase_list_valid)
    agg_rmse_valid = np.sqrt(np.mean(mse_list_valid))
    agg_rel_rmse_valid = np.mean(rel_rmse_list_valid)


    
    print("Validation Metrics:")
    print("sMAPE:", agg_metric_valid["sMAPE"])
    print("MASE (valid):", agg_mase_valid)
    #print("CRPS:", agg_metric_valid["mean_wQuantileLoss"])
    #print("CRPS-Sum:", agg_metric_valid["m_sum_mean_wQuantileLoss"])
    print("RMSE (valid):", agg_rmse_valid)
    print("Relative RMSE (valid):", agg_rel_rmse_valid)


    """

    evaluator_valid = MultivariateEvaluator(
        quantiles=(np.arange(20)/20.0)[1:],
        target_agg_funcs={'sum': np.sum}
    )
    
    agg_metric_valid, item_metrics_valid = evaluator_valid(targets_valid, forecasts_valid, num_series=len(dataset_valid))
    """
    
else:
    assert config.path is not None
    trainnet = estimator.create_training_network(config.device)
    trainnet.load_state_dict(torch.load(config.path))
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, trainnet, config.device)


forecast_it_test, ts_it_test = make_evaluation_predictions(
    dataset=dataset_test,
    predictor=predictor,
    num_samples=100
)

forecasts_test = list(forecast_it_test)
targets_test = list(ts_it_test)

evaluator_test = MultivariateEvaluator(
    quantiles=(np.arange(20)/20.0)[1:],
    target_agg_funcs={'sum': np.sum})
    
agg_metric_test, item_metrics_test = evaluator_test(targets_test, forecasts_test, num_series=len(dataset_test))

forecasts_test = inverse_transform_forecasts(forecasts_test, scaler)
targets_test = inverse_transform_targets(targets_test, scaler)
    

mae_list_test = []
mase_list_test = []
mse_list_test = []
rmse_list_test = []
rel_rmse_list_test = []

for target, forecast in zip(targets_test, forecasts_test):
    forecast_len = len(forecast)
    target = target[:forecast_len]

    mae = np.mean(np.abs(forecast - target))
    mae_list_test.append(mae)

    naive_mae = np.mean(np.abs(np.diff(target)))
    epsilon = 1e-10 
    mase = mae / (naive_mae + epsilon)
    mase_list_test.append(mase)
        #mse -> after RMSE 
    mse = np.mean((forecast - target) ** 2)
    mse_list_test.append(mse)

    rmse_list_test.append(np.sqrt(mse))

    mean_target = np.mean(target)
    rel_rmse = np.sqrt(mse) / (mean_target + epsilon)
    rel_rmse_list_test.append(rel_rmse)

agg_mase_test = np.mean(mase_list_test)
rmse_list_test = np.sqrt(mse_list_test) 
agg_rmse_test = np.mean(rmse_list_test)
agg_rmse_test = np.mean(rmse_list_test)  # 각 RMSE의 평균
agg_rel_rmse_test = np.mean(rel_rmse_list_test) 


    
print("Test Metrics:")
print("sMAPE:", agg_metric_test["sMAPE"])
print("MASE (Test):", agg_mase_test)
#print("CRPS:", agg_metric_test["mean_wQuantileLoss"])
#print("CRPS-Sum:", agg_metric_test["m_sum_mean_wQuantileLoss"])
print("RMSE (Test):", agg_rmse_test)
print("Relative RMSE (Test):", agg_rel_rmse_test)
metrics = {
    'sMAPE': agg_metric_test["sMAPE"],
    "MASE (Test):_re": agg_mase_test,
    #'CRPS': agg_metric_test["mean_wQuantileLoss"],
    #"CRPS-Sum:": agg_metric_test["m_sum_mean_wQuantileLoss"],
    "RMSE (Test)": agg_rmse_test,
    "Relative RMSE (Test)": agg_rel_rmse_test
}

