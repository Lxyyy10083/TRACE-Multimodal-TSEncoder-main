import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.tools import MetricsStore, dtype_map, make_dir_if_not_exists, count_parameters
from src.utils.metrics import forecast_metric
from .base import Tasks
from pdb import set_trace
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from src.utils.tools import EarlyStopping, gather_across_gpus, flatten_nested_list
import pandas as pd
warnings.filterwarnings("ignore")


class ForecastFinetuning(Tasks):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args
        self._build_model()
        self._logged_time_prior_batch = False

    def _unpack_batch_with_time_prior(self, batch):
        """Support both TimeseriesData batches and tuple batches with time priors."""
        if isinstance(batch, (tuple, list)):
            if len(batch) >= 6:
                batch_x, batch_y, text, time_feat, time_feat_weight, domain_id = batch[:6]
            elif len(batch) >= 5:
                batch_x, batch_y, text, time_feat, domain_id = batch[:5]
                time_feat_weight = None
            elif len(batch) == 3:
                batch_x, batch_y, text = batch
                time_feat, time_feat_weight, domain_id = None, None, None
            elif len(batch) == 2:
                batch_x, batch_y = batch
                text, time_feat, time_feat_weight, domain_id = None, None, None, None
            else:
                batch_x = batch[0]
                batch_y, text, time_feat, time_feat_weight, domain_id = None, None, None, None, None
            return batch_x, batch_y, text, time_feat, time_feat_weight, domain_id

        return (
            batch,
            getattr(batch, "forecast", None),
            getattr(batch, "descriptions", None),
            getattr(batch, "time_feat", None),
            getattr(batch, "time_feat_weight", None),
            getattr(batch, "domain_id", None),
        )

    def _move_time_prior_to_device(self, time_feat, time_feat_weight, domain_id):
        """Move optional time prior tensors to the active training device."""
        if time_feat is not None:
            if not torch.is_tensor(time_feat):
                time_feat = torch.as_tensor(time_feat)
            time_feat = time_feat.to(self.device)
        if time_feat_weight is not None:
            if not torch.is_tensor(time_feat_weight):
                time_feat_weight = torch.as_tensor(time_feat_weight)
            time_feat_weight = time_feat_weight.to(self.device)
        if domain_id is not None:
            if not torch.is_tensor(domain_id):
                domain_id = torch.as_tensor(domain_id, dtype=torch.long)
            domain_id = domain_id.to(self.device)
        return time_feat, time_feat_weight, domain_id

    def _log_time_prior_batch_once(self, batch_x, time_feat, domain_id):
        if self._logged_time_prior_batch or time_feat is None or domain_id is None:
            return
        batch_x_shape = (
            batch_x.timeseries.shape if hasattr(batch_x, "timeseries") else batch_x.shape
        )
        print(
            "[ForecastFinetuning] time prior batch: "
            f"batch_x.shape={batch_x_shape}, "
            f"time_feat.shape={time_feat.shape}, "
            f"domain_id.shape={domain_id.shape}"
        )
        self._logged_time_prior_batch = True
    
    def validation(self, data_loader, return_preds: bool = False):
        trues, preds, histories= [], [], []
        loss_list = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, total=len(data_loader)):
                batch_x, batch_y, text, time_feat, time_feat_weight, domain_id = self._unpack_batch_with_time_prior(batch)
                if hasattr(batch_x, "timeseries"):
                    timeseries = batch_x.timeseries.float().to(self.device) #[B, C, L]
                    input_mask = batch_x.input_mask.long().to(self.device) #[B, C, L]
                    forecast = batch_x.forecast.float().to(self.device)  #[B, C, H]
                else:
                    timeseries = batch_x.float().to(self.device)
                    input_mask = torch.ones_like(timeseries, dtype=torch.long)
                    forecast = batch_y.float().to(self.device)
                time_feat, time_feat_weight, domain_id = self._move_time_prior_to_device(time_feat, time_feat_weight, domain_id)
                self._log_time_prior_batch_once(batch_x, time_feat, domain_id)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model(
                        x_enc=timeseries,
                        input_mask=input_mask,
                        mask=None,
                        time_feat=time_feat,
                        time_feat_weight=time_feat_weight,
                        domain_id=domain_id,
                    )

                loss = self.criterion(outputs.forecast, forecast)

                #### get metrics from all GPUs #####
                if self.args.world_size > 1:                    
                    tensor_forecast_loss = torch.tensor(loss, device=self.device)
                    dist.all_reduce(tensor_forecast_loss, op=dist.ReduceOp.SUM)
                    loss = (tensor_forecast_loss / self.args.world_size)
                #### Finish getting metrics from all GPUs #####
                
                loss_list.append(loss.item())
                if return_preds:
                    trues.append(forecast.detach().cpu().numpy())
                    preds.append(outputs.forecast.detach().cpu().numpy())
                    histories.append(timeseries.detach().cpu().numpy())

        forecast_losses = np.array(loss_list)
        average_forecast_loss = np.average(forecast_losses)
        average_losses = {"val_loss": average_forecast_loss}
        self.model.train()
        if self.args.debug and self.args.rank == 0:
            print(f"Evaluation loss: {average_forecast_loss}")
        
        if return_preds and self.args.world_size > 1:
            gathered_trues = gather_across_gpus(trues)
            gathered_preds = gather_across_gpus(preds)
            gathered_histories = gather_across_gpus(histories)

            if self.args.rank == 0:
                trues = np.concatenate(flatten_nested_list(gathered_trues), axis=0)  # [N, C, H]
                preds = np.concatenate(flatten_nested_list(gathered_preds), axis=0)  # [N, C, H]
                histories = np.concatenate(flatten_nested_list(gathered_histories), axis=0)  # [N, C, L]


        if return_preds:
            if self.args.rank == 0:
                return average_losses, (trues, preds, histories)
            else:
                return average_losses, (None, None, None)
        else:
            return average_losses

    def train(self):
        if self.args.rank == 0:
            self.run_name = self.logger.name
            path = os.path.join(self.args.checkpoint_path, self.run_name)
            make_dir_if_not_exists(path, verbose=True)
            self.results_dir = self._create_results_dir(experiment_name="supervised_forecasting")

        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion(loss_type="huber", delta=1.0)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        self._init_lr_scheduler(type=self.args.lr_scheduler_type)
        
        if self.args.model_name == "TraceEncoder":
            self.load_pretrained_ts_encoder(pretraining_task_name="pretraining", do_not_copy_head=True)
    
        self.model.to(self.args.rank)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.args.rank],
            find_unused_parameters=getattr(self.args, "use_temporal_prior", False),
        )
        # self.early_stopping = EarlyStopping(patience=self.args.patience, delta=self.args.delta)
        
        
        opt_steps = 0
        cur_epoch = 0
        best_validation_loss = np.inf
        while cur_epoch < self.args.max_epoch:
            print(f"Epoch {cur_epoch} of {self.args.max_epoch}")
            self.model.train()
            if self.args.distributed and isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(cur_epoch)
                
            for batch in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
                batch_x, batch_y, text, time_feat, time_feat_weight, domain_id = self._unpack_batch_with_time_prior(batch)
                self.optimizer.zero_grad(set_to_none=True)
                if hasattr(batch_x, "timeseries"):
                    timeseries = batch_x.timeseries.float().to(self.device)  #[B, C, L]
                    input_mask = batch_x.input_mask.long().to(self.device)  #[B, C, L]
                    forecast = batch_x.forecast.float().to(self.device)  #[B, C, H]
                else:
                    timeseries = batch_x.float().to(self.device)
                    input_mask = torch.ones_like(timeseries, dtype=torch.long)
                    forecast = batch_y.float().to(self.device)
                time_feat, time_feat_weight, domain_id = self._move_time_prior_to_device(time_feat, time_feat_weight, domain_id)
                self._log_time_prior_batch_once(batch_x, time_feat, domain_id)
                if not self.args.set_input_mask:
                    input_mask = torch.ones_like(input_mask)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model(
                        x_enc=timeseries,
                        input_mask=input_mask,
                        time_feat=time_feat,
                        time_feat_weight=time_feat_weight,
                        domain_id=domain_id,
                    )

                
                loss = self.criterion(outputs.forecast, forecast)
                
                if self.args.debug:
                    print(f"Step {opt_steps} loss: {loss.item()}")
                    
                if self.args.rank == 0:
                    self.logger.log(
                        {
                            "train_loss": loss.item(),
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )


                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                opt_steps = opt_steps + 1

                # Adjust learning rate
                if self.args.lr_scheduler_type == "linearwarmupcosinelr":
                    self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)
                elif self.args.lr_scheduler_type == "onecyclelr":
                    self.lr_scheduler.step()
            
            
            cur_epoch = cur_epoch + 1
            if cur_epoch % self.args.log_interval == 0:
                if self.args.distributed and isinstance(self.val_dataloader.sampler, DistributedSampler):
                    self.val_dataloader.sampler.set_epoch(cur_epoch)
                    
                eval_metrics =self.evaluate_and_log()

                if eval_metrics.val_loss["val_loss"] < best_validation_loss:
                    best_validation_loss = eval_metrics.val_loss["val_loss"]
                    if self.args.rank == 0 and not self.args.debug:
                        self.save_model(self.model, path, None, self.optimizer, self.scaler)
                
                if self.args.distributed and isinstance(self.test_dataloader.sampler, DistributedSampler):
                    self.test_dataloader.sampler.set_epoch(cur_epoch)                
                test_loss, (trues, preds, _) = self.validation(self.test_dataloader, return_preds=True)
                if self.args.rank == 0:
                    metrics = forecast_metric(preds, trues)
                    forecasting_table = pd.DataFrame(
                        data=[self.run_name, self.logger.id, cur_epoch, metrics["mae"], metrics["mse"],metrics["mape"], metrics["smape"], metrics["rmse"]],
                        index=["Model name", "ID", "Epoch", "MAE", "MSE", "MAPE", "sMAPE", "RMSE"]
                    )

                    self.logger.log({"MAE": metrics["mae"],
                                    "MSE": metrics["mse"],
                                    "MAPE": metrics["mape"],
                                    "sMAPE": metrics["smape"],
                                    "RMSE": metrics["rmse"],
                                    "test_loss": test_loss["val_loss"]})
                    # self.save_results(forecasting_table, self.results_dir)
        
        return self.model

    def evaluate_model(self):
        return MetricsStore(val_loss=self.validation(self.val_dataloader))

    def evaluate_and_log(self):
        eval_metrics = self.evaluate_model()
        if self.args.rank == 0:
            self.logger.log(eval_metrics.val_loss)
        return eval_metrics
