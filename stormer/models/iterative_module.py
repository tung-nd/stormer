from typing import Any, List, Tuple, Dict

import torch
from lightning import LightningModule
from torchvision.transforms import transforms

from stormer.models.hub.stormer import Stormer
from stormer.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from stormer.utils.metrics import (
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
)
from stormer.utils.data_utils import CONSTANTS, WEIGHT_DICT


class GlobalForecastIterativeModule(LightningModule):
    def __init__(
        self,
        net: Stormer,
        weighted_loss: bool = True,
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        pretrained_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['net'])
        self.net = net
        
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location='cpu')['state_dict']
            msg = self.load_state_dict(state_dict)
            print(msg)
            
    def set_base_intervals_and_lead_times(self, list_train_intervals, val_lead_times):
        # list_train_intervals: list of base intervals, e.g., [6, 12, 24]
        # val_lead_times: list of target lead times, e.g., [72, 120]
        self.val_lead_times = val_lead_times
        self.list_train_intervals = list_train_intervals

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon
        
    def set_transforms(self, inp_transform, diff_transform):
        self.inp_transform = inp_transform
        self.reverse_inp_transform = self.get_reverse_transform(inp_transform)
        
        self.diff_transform = diff_transform
        self.reverse_diff_transform = {
            k: self.get_reverse_transform(v) for k, v in diff_transform.items()
        }
    
    def get_reverse_transform(self, transform):
        mean, std = transform.mean, transform.std
        std_reverse = 1 / std
        mean_reverse = -mean * std_reverse
        return transforms.Normalize(mean_reverse, std_reverse)
    
    def replace_constant(self, yhat, out_variables):
        for i in range(yhat.shape[1]):
            if out_variables[i] in CONSTANTS:
                yhat[:, i] = 0.0
        return yhat
    
    def pad(self, x: torch.Tensor):
        h = x.shape[-2]
        # Calculate the pad size for the height if it's not divisible by the patch size
        if h % self.net.patch_size != 0:
            pad_size = self.net.patch_size - h % self.net.patch_size
            # Only pad the top
            padded_x = torch.nn.functional.pad(x, (0, 0, pad_size, 0), 'constant', 0)
        else:
            padded_x = x
            pad_size = 0
        return padded_x, pad_size
    
    def forward(self, x: torch.Tensor, variables, interval) -> torch.Tensor:
        padded_x, pad_size = self.pad(x)
        output = self.net(padded_x, variables, interval)[:, :, pad_size:]
        return output
    
    def forward_train(self, x: torch.Tensor, variables, interval_tensors, mean_diff_transform, std_diff_transform):
        # x: initial condition, B, V, H, W
        # variables: list of variable names
        # interval_tensors: B, T, same value along the T dimension (homogeneous) but can be different across the B dimension
        # mean_diff_transform: B, V
        # std_diff_transform: B, V

        norm_diffs = []
        mean_diff_transform = mean_diff_transform.unsqueeze(-1).unsqueeze(-1) # B, V, 1, 1
        std_diff_transform = std_diff_transform.unsqueeze(-1).unsqueeze(-1) # B, V, 1, 1
        n_steps = interval_tensors.shape[-1]

        # x is always in the normalized input space
        for i in range(n_steps):
            norm_pred_diff = self(x, variables, interval_tensors[:, i]) # diff in the normalized space
            norm_pred_diff = self.replace_constant(norm_pred_diff, variables)
            norm_diffs.append(norm_pred_diff)
            raw_pred_diff = norm_pred_diff * std_diff_transform + mean_diff_transform # diff in the original space
            pred = self.reverse_inp_transform(x) + raw_pred_diff # prediction in the original space
            x = self.inp_transform(pred) # prediction in the normalized space
        return norm_diffs

    def training_step(self, batch: Any, batch_idx: int):
        x, gt_diff, mean_diff_transform, std_diff_transform, interval_tensors, variables = batch
        pred_diff = self.forward_train(x, variables, interval_tensors, mean_diff_transform, std_diff_transform)
        pred_diff = torch.stack(pred_diff, dim=1).flatten(0, 1)  # B*T, V, H, W
        gt_diff = gt_diff.flatten(0, 1)  # B*T, V, H, W
        loss_dict = lat_weighted_mse(
            pred_diff,
            gt_diff,
            variables,
            self.lat,
            weighted=self.hparams.weighted_loss,
            weight_dict=WEIGHT_DICT
        )
        
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=x.shape[0],
            )

        return loss_dict[f"w_mse_aggregate"]
    
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
        batch_idx: int,
    ) -> torch.Tensor:
        self.evaluate(batch, self.val_lead_times, "val")
        
    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
        batch_idx: int,
    ) -> torch.Tensor:
        self.evaluate(batch, self.val_lead_times, "test")
    
    def forward_validation(self, x: torch.Tensor, variables, interval, steps):
        # x: initial condition, B, V, H, W
        # variables: list of variable names
        # interval: scalar value, e.g., 6, use the same interval across the batch
        # steps: scalar value, e.g., 24, number of autoregressive steps

        # x is always in the normalized input space
        interval_tensor = torch.Tensor([interval]).to(device=x.device, dtype=x.dtype) / 10.0
        interval_tensor = interval_tensor.repeat(x.shape[0])
        for _ in range(steps):
            pred_diff = self(x, variables, interval_tensor) # diff in the normalized space
            pred_diff = self.replace_constant(pred_diff, variables)
            pred_diff = self.reverse_diff_transform[interval](pred_diff) # diff in the original space
            pred = self.reverse_inp_transform(x) + pred_diff # prediction in the original space
            x = self.inp_transform(pred) # prediction in the normalized space
        return x
    
    def evaluate(
        self, batch: Tuple[torch.Tensor, Dict, List[str], List[str]],
        val_lead_times: List[int],
        stage: str
    ):
        x, dict_y, variables = batch
        
        def get_loss_dict(y, yhat, list_metrics, postfix):
            all_loss_dicts = []
            for metric in list_metrics:
                loss_dict = metric(
                    yhat,
                    y,
                    self.reverse_inp_transform,
                    variables,
                    lat=self.lat,
                    log_postfix=postfix,
                    weighted=self.hparams.weighted_loss,
                    weight_dict=WEIGHT_DICT
                )
                all_loss_dicts.append(loss_dict)
            
            final_loss_dict = {}
            for d in all_loss_dicts:
                final_loss_dict.update(d)
                
            final_loss_dict = {f"{stage}/{k}": v for k, v in final_loss_dict.items()}
            return final_loss_dict
        
        for target_lead_time in val_lead_times:
            all_norm_preds = []
            ### roll-out using a single base interval
            for base_interval in self.list_train_intervals:
                if target_lead_time % base_interval == 0:
                    steps = target_lead_time // base_interval
                    norm_pred = self.forward_validation(x, variables, base_interval, steps)
                    base_loss_dict = get_loss_dict(
                        dict_y[target_lead_time],
                        norm_pred,
                        list_metrics=[lat_weighted_mse_val, lat_weighted_rmse],
                        postfix=f"{target_lead_time}_hrs_base_{base_interval}"
                    )
                    all_norm_preds.append(norm_pred)
                    
                    self.log_dict(
                        base_loss_dict,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=x.shape[0],
                    )
            
            mean_norm_pred = torch.stack(all_norm_preds, dim=0).mean(0) # ensemble mean
            ensemble_loss_dict = get_loss_dict(
                dict_y[target_lead_time],
                mean_norm_pred,
                list_metrics=[lat_weighted_mse_val, lat_weighted_rmse],
                postfix=f"{target_lead_time}_hrs_ensemble_mean"
            )
            
            self.log_dict(
                ensemble_loss_dict,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "channel_embed" in name or "pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0,
                },
            ]
        )

        n_steps_per_machine = len(self.trainer.datamodule.train_dataloader())
        n_steps = int(n_steps_per_machine / (self.trainer.num_devices * self.trainer.num_nodes))
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs * n_steps,
            self.hparams.max_epochs * n_steps,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
