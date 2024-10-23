# Standard library
import os
from typing import Optional, Sequence, Tuple

# Third party
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from lightning import LightningDataModule

# Local application
from stormer.data.iterative_dataset import ERA5OneStepRandomizedDataset, ERA5MultiLeadtimeDataset, ERA5MultiLeadtimeForecastOnlyDataset


def collate_fn_one_step(
    batch,
) -> Tuple[torch.tensor, torch.tensor, Sequence[str], Sequence[str]]:
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, T, C, H, W
    raw_inp = torch.stack([batch[i][1] for i in range(len(batch))]) # B, T, C, H, W
    out = torch.stack([batch[i][2] for i in range(len(batch))]) # B, C, H, W
    interval = torch.cat([batch[i][3] for i in range(len(batch))]) # B
    variables = batch[0][4]
    out_variables = batch[0][5]
    return inp, raw_inp, out, interval, variables, out_variables


def collate_fn_multi_lead_time(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, T, C, H, W
    raw_inp = torch.stack([batch[i][1] for i in range(len(batch))]) # B, T, C, H, W
    
    out_dicts = [batch[i][2] for i in range(len(batch))]
    if out_dicts[0] is None:
        out = None
        raw_out = None
    else:
        list_lead_times = out_dicts[0].keys()
        out = {}
        for lead_time in list_lead_times:
            out[lead_time] = torch.stack([out_dicts[i][lead_time] for i in range(len(batch))])
            
        raw_out_dicts = [batch[i][3] for i in range(len(batch))]
        raw_out = {}
        for lead_time in list_lead_times:
            raw_out[lead_time] = torch.stack([raw_out_dicts[i][lead_time] for i in range(len(batch))])
        
    variables = batch[0][4]
    out_variables = batch[0][5]
    
    return inp, raw_inp, out, raw_out, variables, out_variables


def collate_fn_multi_lead_time_with_time(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, T, C, H, W
    raw_inp = torch.stack([batch[i][1] for i in range(len(batch))]) # B, T, C, H, W
    
    inp_time = [batch[i][2] for i in range(len(batch))]
    
    out_dicts = [batch[i][3] for i in range(len(batch))]
    if out_dicts[0] is None:
        out = None
        raw_out = None
        out_time_dict = None
    else:
        list_lead_times = out_dicts[0].keys()
        out = {}
        for lead_time in list_lead_times:
            out[lead_time] = torch.stack([out_dicts[i][lead_time] for i in range(len(batch))])
            
        raw_out_dicts = [batch[i][4] for i in range(len(batch))]
        raw_out = {}
        for lead_time in list_lead_times:
            raw_out[lead_time] = torch.stack([raw_out_dicts[i][lead_time] for i in range(len(batch))])
            
        out_times = [batch[i][5] for i in range(len(batch))]
        out_time_dict = {}
        for lead_time in list_lead_times:
            out_time_dict[lead_time] = [out_times[i][lead_time] for i in range(len(batch))]
        
    variables = batch[0][6]
    out_variables = batch[0][7]
    
    return inp, raw_inp, inp_time, out, raw_out, out_time_dict, variables, out_variables


class OneStepDataRandomizedModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        variables,
        list_train_intervals,
        val_lead_times,
        data_freq=6,
        year_list=None,
        batch_size=1,
        val_batch_size=2,
        num_workers=0,
        pin_memory=False,
        return_metadata=False,
        gen_forecast_only=False
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # normalization for input
        normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
        normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
        normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
        self.transforms = transforms.Normalize(normalize_mean, normalize_std)
        
        diff_transforms = {}
        for l in list_train_intervals:
            normalize_diff_std = dict(np.load(os.path.join(root_dir, f"normalize_diff_std_{l}.npz")))
            normalize_diff_std = np.concatenate([normalize_diff_std[v] for v in variables], axis=0)
            diff_transforms[l] = transforms.Normalize(np.zeros_like(normalize_diff_std), normalize_diff_std)
        self.diff_transforms = diff_transforms

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npy"))
        return lat, lon
    
    def get_transforms(self):
        return self.transforms, self.diff_transforms

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ERA5OneStepRandomizedDataset(
                root_dir=os.path.join(self.hparams.root_dir, 'train'),
                variables=self.hparams.variables,
                transform=self.transforms,
                dict_diff_transform=self.diff_transforms,
                list_intervals=self.hparams.list_train_intervals,
                data_freq=self.hparams.data_freq,
                year_list=self.hparams.year_list
            )
            
            dataset_cls = ERA5MultiLeadtimeDataset if not self.hparams.gen_forecast_only else ERA5MultiLeadtimeForecastOnlyDataset

            if os.path.exists(os.path.join(self.hparams.root_dir, 'val')):
                self.data_val = dataset_cls(
                    root_dir=os.path.join(self.hparams.root_dir, 'val'),
                    variables=self.hparams.variables,
                    list_lead_times=self.hparams.val_lead_times,
                    transform=self.transforms,
                    data_freq=self.hparams.data_freq,
                    year_list=self.hparams.year_list,
                    return_metadata=self.hparams.return_metadata
                )

            if os.path.exists(os.path.join(self.hparams.root_dir, 'test')):
                self.data_test = dataset_cls(
                    root_dir=os.path.join(self.hparams.root_dir, 'test'),
                    variables=self.hparams.variables,
                    list_lead_times=self.hparams.val_lead_times,
                    transform=self.transforms,
                    data_freq=self.hparams.data_freq,
                    year_list=self.hparams.year_list,
                    return_metadata=self.hparams.return_metadata
                )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_one_step
        )

    def val_dataloader(self):
        if self.data_val is not None:
            return DataLoader(
                self.data_val,
                batch_size=self.hparams.val_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=collate_fn_multi_lead_time
            )

    def test_dataloader(self):
        if self.data_test is not None:
            return DataLoader(
                self.data_test,
                batch_size=self.hparams.val_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=collate_fn_multi_lead_time if not self.hparams.return_metadata else collate_fn_multi_lead_time_with_time
            )

# datamodule = OneStepDataModule(
#     '/eagle/MDClimSim/tungnd/data/wb1/1.40625deg_1_step_6hr',
#     variables=[
#         "land_sea_mask",
#         "orography",
#         "lattitude",
#         "2m_temperature",
#         "10m_u_component_of_wind",
#         "10m_v_component_of_wind",
#         "toa_incident_solar_radiation",
#         "total_cloud_cover",
#         "geopotential_500",
#         "temperature_850"
#     ],
#     batch_size=128,
#     num_workers=1,
#     pin_memory=False
# )
# datamodule.setup()
# for batch in datamodule.train_dataloader():
#     inp, out, vars, out_vars = batch
#     print (inp.shape)
#     print (out.shape)
#     print (vars)
#     print (out_vars)
#     break