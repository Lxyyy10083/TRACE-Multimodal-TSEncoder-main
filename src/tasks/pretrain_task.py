import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from wandb import AlertLevel
from src.utils.tools import MetricsStore, dtype_map, make_dir_if_not_exists, count_parameters
from .base import Tasks
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from src.utils.tools import EarlyStopping
warnings.filterwarnings("ignore")


class Pretraining(Tasks):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args
        self._build_model()
        count_parameters(self.model)

    def validation(self, data_loader, return_preds: bool = False, split: str = "val"):
        loss = {"forecast_losses": [], "total_losses": []}

        self.model.eval()
        with torch.no_grad():
            for batch_x in tqdm(data_loader, total=len(data_loader)):
                timeseries = batch_x.timeseries.float().to(self.device) #[B, C, L]
                input_mask = batch_x.input_mask.long().to(self.device) #[B, C, L]

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=None
                    )

                recon_loss = self.forecast_criterion(outputs.reconstruction, timeseries)  #[B, C, L]
                # compute loss on (input_mask = 1 & pre-train_mask = 0)
                observed_mask = input_mask * (1 - outputs.pretrain_mask)  #[B, C, L]
                masked_loss = observed_mask * recon_loss
                forecast_loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)
                total_loss = forecast_loss

                #### get metrics from all GPUs #####
                if self.args.world_size > 1:
                    tensor_forecast_loss = forecast_loss.detach().clone()
                    dist.all_reduce(tensor_forecast_loss, op=dist.ReduceOp.SUM)
                    forecast_loss = (tensor_forecast_loss / self.args.world_size)
                    total_loss = forecast_loss


                loss["forecast_losses"].append(forecast_loss.item())
                loss["total_losses"].append(total_loss.item())

        average_total_loss = np.average(np.array(loss["total_losses"]))
        average_forecast_loss = np.average(np.array(loss["forecast_losses"]))
        average_losses = {f"{split}_total_loss": average_total_loss, f"{split}_recon_loss": average_forecast_loss}
        self.model.train()
        return average_losses

    def train(self):
        if self.args.rank == 0:
            self.run_name = self.logger.name
            path = os.path.join(self.args.checkpoint_path, self.run_name)
            make_dir_if_not_exists(path, verbose=True)

        self.optimizer = self._select_optimizer()
        self.forecast_criterion = self._select_criterion()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        self._init_lr_scheduler()
        self.model.to(self.device)

        if self.args.distributed and self.args.world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args.rank],
                find_unused_parameters=True
            )
        self.early_stopping = EarlyStopping(patience=self.args.patience, delta=self.args.delta)
        # self.evaluate_model()


        opt_steps = 0
        cur_epoch = 0
        while cur_epoch < self.args.max_epoch:
            print(f"Epoch {cur_epoch} of {self.args.max_epoch}")
            self.model.train()
            if self.args.distributed and isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(cur_epoch)

            for batch_x in tqdm(
                self.train_dataloader, total=len(self.train_dataloader)
            ):
                self.optimizer.zero_grad(set_to_none=True)
                timeseries = batch_x.timeseries.float().to(self.device)  #[B, C, L]
                input_mask = batch_x.input_mask.long().to(self.device)  #[B, C, L]

                if not self.args.set_input_mask:
                    input_mask = torch.ones_like(input_mask)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model(x_enc=timeseries, input_mask=input_mask)

                recon_loss = self.forecast_criterion(outputs.reconstruction, timeseries)  #[B, C, L]
                observed_mask = input_mask * (1 - outputs.pretrain_mask)  #[B, C, L]
                masked_loss = observed_mask * recon_loss  #[B, C, L]
                recon_loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)  #[B, C, L]
                total_loss = recon_loss

                if self.args.rank == 0:
                    self.logger.log(
                        {
                            "train_total_loss": total_loss.item(),
                            "train_recon_loss": recon_loss.item(),
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

                if self.args.debug and opt_steps >= 1:
                    self.debug_model_outputs(total_loss, outputs, batch_x)

                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                opt_steps = opt_steps + 1

                self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)

            if cur_epoch % self.args.log_interval == 0:
                if self.args.distributed and isinstance(self.val_dataloader.sampler, DistributedSampler):
                    self.val_dataloader.sampler.set_epoch(cur_epoch)
                    self.val_dataloader.sampler.set_epoch(cur_epoch)
                eval_metrics =self.evaluate_and_log()

                if self.args.rank == 0 and self.args.early_stop == True:
                    self.early_stopping(eval_metrics.val_loss["val_total_loss"], self.model, path=os.path.join(path, "best_checkpoint.pth"))

                stop_signal = torch.tensor(float(self.early_stopping.early_stop), device=self.device)
                dist.broadcast(stop_signal, src=0)

                if stop_signal.item() > 0:
                    break

            #### save checkpoints every epoch #####
            if cur_epoch % self.args.checkpoint_interval == 0:
                ## save checkpoints every epoch
                if self.args.rank == 0:
                    self.logger.alert(
                        title="Saving model",
                        text=f"Saving model after {opt_steps} steps",
                        level=AlertLevel.INFO,
                        )
                    self.save_model(model=self.model, path=path, opt_steps=None, optimizer=self.optimizer, scaler=self.scaler)
            ########################################

            cur_epoch = cur_epoch + 1

        return self.model

    def evaluate_and_log(self):
        eval_metrics = MetricsStore(val_loss=self.validation(self.val_dataloader, split="val"), test_loss=self.validation(self.test_dataloader, split="test"))
        if self.args.rank == 0:
            self.logger.log(eval_metrics.val_loss)
            self.logger.log(eval_metrics.test_loss)
        return eval_metrics
