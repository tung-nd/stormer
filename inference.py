import os
import numpy as np
import torch
from torchvision.transforms import transforms
from stormer.data.iterative_dataset import ERA5MultiLeadtimeDataset
from stormer.models.iterative_module import GlobalForecastIterativeModule
from stormer.models.hub.stormer import Stormer

variables = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential_50",
    "geopotential_100",
    "geopotential_150",
    "geopotential_200",
    "geopotential_250",
    "geopotential_300",
    "geopotential_400",
    "geopotential_500",
    "geopotential_600",
    "geopotential_700",
    "geopotential_850",
    "geopotential_925",
    "geopotential_1000",
    "u_component_of_wind_50",
    "u_component_of_wind_100",
    "u_component_of_wind_150",
    "u_component_of_wind_200",
    "u_component_of_wind_250",
    "u_component_of_wind_300",
    "u_component_of_wind_400",
    "u_component_of_wind_500",
    "u_component_of_wind_600",
    "u_component_of_wind_700",
    "u_component_of_wind_850",
    "u_component_of_wind_925",
    "u_component_of_wind_1000",
    "v_component_of_wind_50",
    "v_component_of_wind_100",
    "v_component_of_wind_150",
    "v_component_of_wind_200",
    "v_component_of_wind_250",
    "v_component_of_wind_300",
    "v_component_of_wind_400",
    "v_component_of_wind_500",
    "v_component_of_wind_600",
    "v_component_of_wind_700",
    "v_component_of_wind_850",
    "v_component_of_wind_925",
    "v_component_of_wind_1000",
    "temperature_50",
    "temperature_100",
    "temperature_150",
    "temperature_200",
    "temperature_250",
    "temperature_300",
    "temperature_400",
    "temperature_500",
    "temperature_600",
    "temperature_700",
    "temperature_850",
    "temperature_925",
    "temperature_1000",
    "specific_humidity_50",
    "specific_humidity_100",
    "specific_humidity_150",
    "specific_humidity_200",
    "specific_humidity_250",
    "specific_humidity_300",
    "specific_humidity_400",
    "specific_humidity_500",
    "specific_humidity_600",
    "specific_humidity_700",
    "specific_humidity_850",
    "specific_humidity_925",
    "specific_humidity_1000",
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load pretrained model
net = Stormer(
    in_img_size=[128, 256],
    variables=variables,
    patch_size=2,
    hidden_size=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4,
)
pretrained_path = 'https://huggingface.co/tungnd/stormer/resolve/main/stormer_1.40625_patch_size_2.ckpt'
model = GlobalForecastIterativeModule(net, pretrained_path=pretrained_path).to(device)
model.eval()

# load data
root_dir = '/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df'
normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
dataset = ERA5MultiLeadtimeDataset(
    root_dir=os.path.join(root_dir, 'test'),
    variables=variables,
    transform=transforms.Normalize(normalize_mean, normalize_std),
    list_lead_times=[24, 72, 120, 168],  # 1, 3, 5, 7 days
    data_freq=6,
)
inp_data, out_data_dict, _ = dataset[0]
inp_data = inp_data.unsqueeze(0).to(device)
out_data_dict = {k: v.unsqueeze(0).to(device) for k, v in out_data_dict.items()}

prediction_dict = {}
list_intervals = [6, 12, 24]
for lead_time in out_data_dict.keys():
    all_preds = []
    for interval in list_intervals:
        if lead_time % interval == 0:
            steps = lead_time // interval
            pred = model.forward_validation(inp_data, variables, interval, steps)
            all_preds.append(pred)
    mean_pred = torch.stack(all_preds, dim=0).mean(0) # ensemble mean
    prediction_dict[lead_time] = mean_pred

# compute metrics here
# ...