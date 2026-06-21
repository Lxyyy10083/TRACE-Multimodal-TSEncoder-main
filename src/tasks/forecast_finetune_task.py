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
        self._logged_forecast_shapes = False
        self.use_direct_text_forecast = (
            getattr(args, "use_direct_text_forecast", False)
            and not getattr(args, "ts_only", False)
        )
        self._direct_text_log_batches = 0
        self._logged_direct_text_shapes = False

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

    def _get_direct_text_batch(self, batch_x):
        text_emb = getattr(batch_x, "text_emb", None)
        text_mask = getattr(batch_x, "text_mask", None)
        if not self.use_direct_text_forecast or text_emb is None or text_mask is None:
            return None, None
        return text_emb.float().to(self.device), text_mask.float().to(self.device)

    def _log_text_leakage_batch(self, batch_x):
        if self.args.rank != 0 or self._direct_text_log_batches >= 3:
            return
        origins = getattr(batch_x, "forecast_origin_time", None)
        selected_times = getattr(batch_x, "text_time", None)
        if origins is None or selected_times is None:
            return
        origin = origins[0]
        selected = selected_times[0]
        passed = selected is None or selected <= origin
        print(
            "[Direct text leakage] "
            f"forecast_origin_time={origin}, "
            f"selected_text_time={selected}, passed={passed}"
        )
        if getattr(self.args, "use_text_leakage_check", True) and not passed:
            raise RuntimeError(
                f"Text leakage detected: {selected} > {origin}"
            )
        self._direct_text_log_batches += 1

    def _log_direct_text_shapes_once(self, outputs, text_emb, text_mask):
        if self._logged_direct_text_shapes or self.args.rank != 0:
            return
        print(
            "[Direct text forecast] "
            f"use_direct_text_forecast={self.use_direct_text_forecast}, "
            f"text_data_path={getattr(self.args, 'text_data_path', None)}, "
            f"text_embedding_path={getattr(self.args, 'text_embedding_path', None)}"
        )
        if self.use_direct_text_forecast:
            print(f"text_emb shape={None if text_emb is None else text_emb.shape}")
            print(
                "text_mask sum="
                f"{None if text_mask is None else text_mask.sum().item()}"
            )
            print(f"ts_emb shape={None if outputs.cls_embedding is None else outputs.cls_embedding.shape}")
            print(f"time_emb shape={None if outputs.time_emb is None else outputs.time_emb.shape}")
            print(f"fused_emb shape={None if outputs.fused_emb is None else outputs.fused_emb.shape}")
            gate = outputs.direct_text_gate
            if gate is not None:
                print(
                    "gate mean/min/max="
                    f"{gate.mean().item():.6f}/"
                    f"{gate.min().item():.6f}/"
                    f"{gate.max().item():.6f}"
                )
        self._logged_direct_text_shapes = True

    def _validate_forecast_shapes(
        self,
        outputs,
        forecast,
        time_feat=None,
        time_feat_weight=None,
    ):
        """Reject broadcasting and verify that all future tensors use H steps."""
        pred = outputs.forecast
        if pred is None:
            raise ValueError("Model output does not contain outputs.forecast")
        if pred.shape != forecast.shape:
            raise ValueError(
                f"Forecast shape mismatch: pred={pred.shape}, target={forecast.shape}"
            )
        if pred.ndim != 3:
            raise ValueError(
                f"Forecast tensors must be [B, C, H], got {pred.shape}"
            )

        horizon = self.args.forecast_horizon
        if pred.shape[-1] != horizon:
            raise ValueError(
                f"Forecast horizon mismatch: configured={horizon}, got={pred.shape[-1]}"
            )

        for name, tensor in (
            ("time_feat", time_feat),
            ("time_feat_weight", time_feat_weight),
        ):
            if tensor is not None and (
                tensor.ndim != 3
                or tensor.shape[0] != pred.shape[0]
                or tensor.shape[1] != horizon
            ):
                raise ValueError(
                    f"{name} must be [B, H, D] with H={horizon}, got {tensor.shape}"
                )

        if not self._logged_forecast_shapes:
            if self.args.rank == 0:
                print(f"outputs.forecast.shape = {list(pred.shape)}")
                print(f"forecast.shape = {list(forecast.shape)}")
                print(
                    "time_feat.shape = "
                    f"{None if time_feat is None else list(time_feat.shape)}"
                )
                print(
                    "time_feat_weight.shape = "
                    f"{None if time_feat_weight is None else list(time_feat_weight.shape)}"
                )
            self._logged_forecast_shapes = True

    def _save_best_checkpoint(self, checkpoint_path, epoch, val_loss):
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_epoch": int(epoch),
                "best_validation_loss": float(val_loss),
            },
            checkpoint_path,
        )

    def _load_best_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
    
    def validation(self, data_loader, return_preds: bool = False):
        trues, preds, histories= [], [], []
        loss_sum = 0.0
        sample_count = 0
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
                text_emb, text_mask = self._get_direct_text_batch(batch_x)
                self._log_text_leakage_batch(batch_x)

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
                        text_emb=text_emb,
                        text_mask=text_mask,
                    )

                self._log_direct_text_shapes_once(outputs, text_emb, text_mask)
                self._validate_forecast_shapes(
                    outputs,
                    forecast,
                    time_feat,
                    time_feat_weight,
                )
                # Aurora-compatible evaluation: MSE in standardized space.
                # Do not inverse_transform predictions or targets here.
                loss = torch.mean((outputs.forecast - forecast) ** 2)
                batch_size = forecast.shape[0]
                loss_sum += loss.item() * batch_size
                sample_count += batch_size
                if return_preds:
                    trues.append(forecast.detach().cpu().numpy())
                    preds.append(outputs.forecast.detach().cpu().numpy())
                    histories.append(timeseries.detach().cpu().numpy())

        loss_stats = torch.tensor(
            [loss_sum, sample_count],
            dtype=torch.float64,
            device=self.device,
        )
        if self.args.world_size > 1:
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
        average_forecast_loss = (loss_stats[0] / loss_stats[1].clamp_min(1)).item()
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

        elif return_preds and self.args.rank == 0:
            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            histories = np.concatenate(histories, axis=0)


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
        if self.args.world_size > 1:
            run_name_holder = [self.run_name if self.args.rank == 0 else None]
            dist.broadcast_object_list(run_name_holder, src=0)
            self.run_name = run_name_holder[0]

        path = os.path.join(self.args.checkpoint_path, self.run_name)
        best_checkpoint_path = os.path.join(path, "best_checkpoint.pth")
        if self.args.rank == 0:
            make_dir_if_not_exists(path, verbose=True)
            self.results_dir = self._create_results_dir(experiment_name="supervised_forecasting")
        if self.args.world_size > 1:
            dist.barrier()

        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion(loss_type="mse")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        self._init_lr_scheduler(type=self.args.lr_scheduler_type)
        
        if self.args.model_name == "TraceEncoder":
            self.load_pretrained_ts_encoder(pretraining_task_name="pretraining", do_not_copy_head=True)
            if self.use_direct_text_forecast:
                for name, parameter in self.model.named_parameters():
                    if "direct_text_fusion" in name:
                        parameter.requires_grad = True
    
        self.model.to(self.device)
        if self.args.world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args.rank],
                find_unused_parameters=getattr(
                    self.args,
                    "use_temporal_prior",
                    False,
                ),
            )
            if self.args.rank == 0:
                print("[DDP] enabled for forecast_finetune")
        elif self.args.rank == 0:
            print("[DDP] skipped because world_size=1")
        # self.early_stopping = EarlyStopping(patience=self.args.patience, delta=self.args.delta)
        
        
        opt_steps = 0
        cur_epoch = 0
        best_validation_loss = np.inf
        best_epoch = None
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
                text_emb, text_mask = self._get_direct_text_batch(batch_x)
                self._log_text_leakage_batch(batch_x)
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
                        text_emb=text_emb,
                        text_mask=text_mask,
                    )

                self._log_direct_text_shapes_once(outputs, text_emb, text_mask)
                self._validate_forecast_shapes(
                    outputs,
                    forecast,
                    time_feat,
                    time_feat_weight,
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
            should_validate = (
                cur_epoch % self.args.log_interval == 0
                or cur_epoch == self.args.max_epoch
            )
            if should_validate:
                if self.args.distributed and isinstance(self.val_dataloader.sampler, DistributedSampler):
                    self.val_dataloader.sampler.set_epoch(cur_epoch)
                    
                eval_metrics = self.evaluate_and_log()
                current_val_loss = eval_metrics.val_loss["val_loss"]

                if current_val_loss < best_validation_loss:
                    best_validation_loss = current_val_loss
                    best_epoch = cur_epoch
                    if self.args.rank == 0:
                        self._save_best_checkpoint(
                            best_checkpoint_path,
                            best_epoch,
                            best_validation_loss,
                        )

        if best_epoch is None:
            raise RuntimeError("No validation result was produced; best checkpoint is unavailable")
        if self.args.world_size > 1:
            dist.barrier()

        best_checkpoint = self._load_best_checkpoint(best_checkpoint_path)
        best_epoch = int(best_checkpoint["best_epoch"])
        best_validation_loss = float(best_checkpoint["best_validation_loss"])

        if self.args.distributed and isinstance(self.test_dataloader.sampler, DistributedSampler):
            self.test_dataloader.sampler.set_epoch(best_epoch)
        test_loss, (trues, preds, _) = self.validation(
            self.test_dataloader,
            return_preds=True,
        )
        if self.args.rank == 0:
            # MMDataset already standardized these arrays with the train-fitted
            # scaler. Deliberately do not inverse_transform before metrics.
            metrics = forecast_metric(preds, trues)
            print("Final Forecast Metrics")
            print(f"Best epoch: {best_epoch}")
            print(f"Best validation loss: {best_validation_loss:.6f}")
            print(f"Test MSE: {metrics['mse']:.6f}")
            print(f"Test MAE: {metrics['mae']:.6f}")
            print(f"Test RMSE: {metrics['rmse']:.6f}")
            self.logger.log(
                {
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "test_MAE": metrics["mae"],
                    "test_MSE": metrics["mse"],
                    "test_MAPE": metrics["mape"],
                    "test_sMAPE": metrics["smape"],
                    "test_RMSE": metrics["rmse"],
                    "test_loss": test_loss["val_loss"],
                }
            )
        
        return self.model

    def evaluate_model(self):
        return MetricsStore(val_loss=self.validation(self.val_dataloader))

    def evaluate_and_log(self):
        eval_metrics = self.evaluate_model()
        if self.args.rank == 0:
            self.logger.log(eval_metrics.val_loss)
        return eval_metrics
