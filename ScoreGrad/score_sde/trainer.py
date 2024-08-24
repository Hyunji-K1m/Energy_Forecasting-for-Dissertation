import time
from typing import List, Optional, Union

from tqdm import tqdm
#import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from gluonts.core.component import validated
from score_sde.ema import ExponentialMovingAverage


class Trainer:
    @validated()
    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 8,
        num_batches_per_epoch: int = 30, #change
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        maximum_learning_rate: float = 1e-3,
        wandb_mode: str = "disabled",
        clip_gradient: float=1.0, #Optional[float] = None,
        decay=0.999,
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.clip_gradient = clip_gradient
        # self.decay = decay
        self.device = device
        #wandb.init(mode=wandb_mode, **kwargs)

    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:
        net.to(self.device)
        #if hasattr(wandb, 'watch'):
            #wandb.watch(net, log="all", log_freq=self.num_batches_per_epoch)
        # if self.decay is not None:
        #     ema = ExponentialMovingAverage(net.parameters(), self.decay)

        #optimizer = Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(net.parameters(), lr=self.learning_rate, momentum=0.9)

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.maximum_learning_rate,
            steps_per_epoch=self.num_batches_per_epoch,
            epochs=self.epochs,
        )

        for epoch_no in range(self.epochs):
            # mark epoch start time
            tic = time.time()
            avg_epoch_loss = 0.0
            with tqdm(train_iter) as it:
                for batch_no, data_entry in enumerate(it, start=1):
                    optimizer.zero_grad()
                    inputs = [v.to(self.device) for v in data_entry.values()]

                    output = net(*inputs)
                    if isinstance(output, (list, tuple)):
                        loss = output[0]
                    else:
                        loss = output

                    avg_epoch_loss += loss.item()
                    it.set_postfix(
                        ordered_dict={
                            "avg_epoch_loss": avg_epoch_loss / batch_no,
                            "epoch": epoch_no,
                        },
                        refresh=False,
                    )
                    #wandb.log({"loss": loss.item()})

                    loss.backward()
                    if self.clip_gradient is not None:
                        nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)

                    optimizer.step()
                    lr_scheduler.step()
                    # if self.decay is not None:
                    #     ema.update(net.parameters()) 

                    if self.num_batches_per_epoch == batch_no:
                        break

            # mark epoch end time and log time cost of current epoch
            toc = time.time()

            if validation_iter is not None:
                net.eval()  # Set model to evaluation mode
                avg_val_loss = 0.0
                with torch.no_grad():  # Disable gradient calculation
                    with tqdm(validation_iter, desc=f"Validation epoch {epoch_no+1}") as val_it:
                        for val_batch_no, val_data_entry in enumerate(val_it, start=1):
                            val_inputs = [v.to(self.device) for v in val_data_entry.values()]
                            val_output = net(*val_inputs)

                            if isinstance(val_output, (list, tuple)):
                                val_loss = val_output[0]
                            else:
                                val_loss = val_output

                            avg_val_loss += val_loss.item()
                            val_it.set_postfix(
                                ordered_dict={
                                    "avg_val_loss": avg_val_loss / val_batch_no,
                                },
                                refresh=False,
                            )
                print(f"Epoch {epoch_no+1}: Avg Validation Loss = {avg_val_loss / len(validation_iter)}")
                net.train()  # Set model back to training mode
