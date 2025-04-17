import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()
import os
import shutil
import zipfile
import tarfile
import urllib
import requests
from tqdm import tqdm
import torch
import timeit

from weaver.utils.logger import _logger
import torch.optim as optim
import time

import torchao
import time

import subprocess

import datetime
curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def print_nvidia_smi():
    # Run the nvidia-smi command and capture the output
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Decode and print the output
    print(result.stdout.decode())

# Example usage
# print_nvidia_smi()

# from dataclasses import field
# common: CommonConfig = field(default_factory=CommonConfig)

# from weaver.nn.model.ParticleTransformer import ParticleTransformer
from ParticleTransformer_Attention_quantizable import ParticleTransformer
# from ParticleTransformer_updated_quant_weights import ParticleTransformer
# 

# torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
#print('cpu')
# device = torch.device("cpu")
print(device)

def build_features_and_labels(tree, transform_features=True):
    # load arrays from the tree
    a = tree.arrays(filter_name=['part_*', 'jet_pt', 'jet_energy', 'label_*'])

    # compute new features
    a['part_mask'] = ak.ones_like(a['part_energy'])
    a['part_pt'] = np.hypot(a['part_px'], a['part_py'])
    a['part_pt_log'] = np.log(a['part_pt'])
    a['part_e_log'] = np.log(a['part_energy'])
    a['part_logptrel'] = np.log(a['part_pt']/a['jet_pt'])
    a['part_logerel'] = np.log(a['part_energy']/a['jet_energy'])
    a['part_deltaR'] = np.hypot(a['part_deta'], a['part_dphi'])
    a['part_d0'] = np.tanh(a['part_d0val'])
    a['part_dz'] = np.tanh(a['part_dzval'])

    # apply standardization
    if transform_features:
        a['part_pt_log'] = (a['part_pt_log'] - 1.7) * 0.7
        a['part_e_log'] = (a['part_e_log'] - 2.0) * 0.7
        a['part_logptrel'] = (a['part_logptrel'] - (-4.7)) * 0.7
        a['part_logerel'] = (a['part_logerel'] - (-4.7)) * 0.7
        a['part_deltaR'] = (a['part_deltaR'] - 0.2) * 4.0
        a['part_d0err'] = _clip(a['part_d0err'], 0, 1)
        a['part_dzerr'] = _clip(a['part_dzerr'], 0, 1)

    feature_list = {
        'pf_points': ['part_deta', 'part_dphi'], # not used in ParT
        'pf_features': [
            'part_pt_log',
            'part_e_log',
            'part_logptrel',
            'part_logerel',
            'part_deltaR',
            'part_charge',
            'part_isChargedHadron',
            'part_isNeutralHadron',
            'part_isPhoton',
            'part_isElectron',
            'part_isMuon',
            'part_d0',
            'part_d0err',
            'part_dz',
            'part_dzerr',
            'part_deta',
            'part_dphi',
        ],
        'pf_vectors': [
            'part_px',
            'part_py',
            'part_pz',
            'part_energy',
        ],
        'pf_mask': ['part_mask']
    }

    out = {}
    for k, names in feature_list.items():
        out[k] = np.stack([_pad(a[n], maxlen=128).to_numpy() for n in names], axis=1)   

    label_list = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']
    out['label'] = np.stack([a[n].to_numpy().astype('int') for n in label_list], axis=1)

    return out

def _clip(a, a_min, a_max):
    try:
        return np.clip(a, a_min, a_max)
    except ValueError:
        return ak.unflatten(np.clip(ak.flatten(a), a_min, a_max), ak.num(a))

def _pad(a, maxlen, value=0, dtype='float32'):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x
    
    
# Select the data directory
tree = uproot.open('/lus/eagle/projects/cms_l1t_fm/quantization/JetClass_example_100k.root')['tree']

table = build_features_and_labels(tree)
    
# creating data

x_particles = table['pf_features']
x_jets = table['pf_vectors']
y = table['label']
x_points = table['pf_points']
x_mask = table['pf_mask']

r_indexes = np.arange(len(x_particles))
np.random.shuffle(r_indexes)

# # train (uncomment if you want to train the model)
# a = 10000
# x_particles_train=x_particles[r_indexes][0:a]
# x_jets_train=x_jets[r_indexes][0:a]
# y_train=y[r_indexes][0:a]
# x_points_train=x_points[r_indexes][0:a]
# x_mask_train=x_mask[r_indexes][0:a]

# test
a = int(32*8*16*2)
x_part_test=x_particles[r_indexes][20000:20000 + a]
x_jet_test=x_jets[r_indexes][20000:20000 + a]
y_test=y[r_indexes][20000:20000 + a]
x_points_test=x_points[r_indexes][20000:20000 + a]
x_mask_test=x_mask[r_indexes][20000:20000 + a]

# sample data for inference

inp = torch.from_numpy(x_points_test),torch.from_numpy(x_part_test),torch.from_numpy(x_jet_test),torch.from_numpy(x_mask_test)

from torch.utils.data import DataLoader

batch = int(32*8*2*10/8)

num_workers = min(os.cpu_count(), 8)

# # Updated DataLoader with optimizations
# dataloader_test = DataLoader(x_part_test, batch_size=batch, shuffle=False, sampler=None,
#            batch_sampler=None, persistent_workers=True, num_workers=num_workers, collate_fn=None,
#            pin_memory=True, drop_last=False, timeout=0,
#            worker_init_fn=None)

# ydataloader_test = DataLoader(y_test, batch_size=batch, shuffle=False, sampler=None,
#            batch_sampler=None, persistent_workers=True, num_workers=num_workers, collate_fn=None,
#            pin_memory=True, drop_last=False, timeout=0,
#            worker_init_fn=None)

# xjdataloader_test = DataLoader(x_jet_test, batch_size=batch, shuffle=False, sampler=None,
#            batch_sampler=None, persistent_workers=True, num_workers=num_workers, collate_fn=None,
#            pin_memory=True, drop_last=False, timeout=0,
#            worker_init_fn=None)

# xpointloader_test = DataLoader(x_points_test, batch_size=batch, shuffle=False, sampler=None,
#            batch_sampler=None, persistent_workers=True, num_workers=num_workers, collate_fn=None,
#            pin_memory=True, drop_last=False, timeout=0,
#            worker_init_fn=None)

# xmaskloader_test = DataLoader(x_mask_test, batch_size=batch, shuffle=False, sampler=None,
#            batch_sampler=None, persistent_workers=True, num_workers=num_workers, collate_fn=None,
#            pin_memory=True, drop_last=False, timeout=0,
#            worker_init_fn=None)

# # old one
dataloader_test = DataLoader(x_part_test, batch_size=batch, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
ydataloader_test = DataLoader(y_test, batch_size=batch, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
xjdataloader_test = DataLoader(x_jet_test, batch_size=batch, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
xpointloader_test = DataLoader(x_points_test, batch_size=batch, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
xmaskloader_test = DataLoader(x_mask_test, batch_size=batch, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

#setting the loss function

loss_fn = torch.nn.CrossEntropyLoss()

# Actual ParticleTransformer model

class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)
        self.attention_matrix = None
        self.interactionMatrix = None
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        output = self.mod(features, v=lorentz_vectors, mask=mask)
        #self.attention_matrix = self.mod.getAttention()
        #self.interactionMatrix = self.mod.getInteraction()
        return output

    def get_attention_matrix(self):
        return self.attention_matrix
    def get_interactionMatrix(self):
        return self.interactionMatrix

# me

def get_model(**kwargs):

    cfg = dict(
        input_dim=17,
        num_classes=10,
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims= [64,64,64],
        # num_heads=8,
        # num_layers=8,
        # embed_dims=[128, 2048, 1024],
        # pair_embed_dims= [256,256,256],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerWrapper(**cfg)

    model_info = {
    }

    return model, model_info

base_model, _ = get_model()

# base_model

# ------------------------------------------------------------------------------------

# # # loading weights in the Actual ParticleTransformer model

# # Load the pretrained weights from the .pt file
# pretrained_dict = torch.load("/lus/eagle/projects/cms_l1t_fm/efficientTransformer/efficient_particle_transformer/attentionWeights/efficient_particle_transformer/models/ParT_full.pt", map_location=torch.device('cpu'))
# #print('pretrained_dict', pretrained_dict.keys())
# # Load only the parameters that exist in the model
# model_dict = base_model.state_dict()
# #print('model_dict', model_dict.keys())

# #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}    # maybe change
# model_dict.update(pretrained_dict)
# #print('model_dict', model_dict.keys())

# base_model.load_state_dict(model_dict)
# # #print('pre_trained_model final', base_model.state_dict().keys())

# # # Set the model to evaluation mode
# # pre_trained_model.eval()

# # print(base_model)
# # print_nvidia_smi()

# ------------------------------------------------------------------------------------

initial = 2

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

from torch.profiler import profile, record_function, ProfilerActivity

print(base_model)
base_model.eval()
base_model.to(device)
print_size_of_model(base_model)

losses = []
execution_time = []

ab = time.time()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'/lus/eagle/projects/cms_l1t_fm/quantization/profile_runs/logs/{curr_time}/profile1'),
    record_shapes=True,
    profile_memory=True
) as prof:
    for i in range(10):
        with torch.no_grad():
            for x,y,z,a,b in zip(dataloader_test, ydataloader_test, xjdataloader_test,xpointloader_test,xmaskloader_test):
                start = time.time()
                y_pred = base_model(a.float().to(device), x.float().to(device), z.float().to(device), b.float().to(device))
                end = time.time()
                losses.append(loss_fn(y_pred, y.float().to(device)).item())
                #print('times', start, end)
                execution_time.append(end-start)

        prof.step()
        
ba = time.time()
print('Total Time', ba-ab)

print('test loss', sum(losses)/len(losses))
print('test execution time', sum(execution_time[initial:])/len(execution_time[initial:]))
print(execution_time[initial:])

# for name, param in base_model.named_parameters():
#     print(f"{name}: {param.dtype}")  # Should print torch.qint8 or torch.int8

#################################################################


import copy
quantizable_model = copy.deepcopy(base_model)

del(base_model)
print_nvidia_smi()

from torchao.quantization import (
    quantize_,
    int8_weight_only,
    int8_dynamic_activation_int8_weight,
    float8_weight_only,
    int4_weight_only
)

# For int 4 quantization
# dtype = torch.bfloat16
# m = ToyLinearModel(1024, 1024, 1024).eval().to(dtype).to("cuda")
# m_bf16 = copy.deepcopy(m)
# example_inputs = m.example_inputs(dtype=dtype, device="cuda")

# m_bf16 = torch.compile(m_bf16, mode='max-autotune')


# dtype = torch.bfloat16
# quantizable_model = quantizable_model.eval().to(dtype).to(device)
# quantizable_model = torch.compile(quantizable_model, mode='max-autotune')
# group_size = 32


quantize_(quantizable_model, int8_weight_only())
#quantize_(quantizable_model, int4_weight_only(group_size=group_size))

#print(id(base_model), id(quantizable_model))

print(quantizable_model)


losses_quant = []
execution_time_quant = []

quantizable_model.eval()
quantizable_model.to(device)
print_size_of_model(quantizable_model)

ab = time.time()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'/lus/eagle/projects/cms_l1t_fm/quantization/profile_runs/logs/{curr_time}/profile2'),
    record_shapes=True,
    profile_memory=True
) as prof:
    for i in range(10):
        with torch.no_grad():
            for x,y,z,a,b in zip(dataloader_test, ydataloader_test, xjdataloader_test,xpointloader_test,xmaskloader_test):
                start = time.time()
                y_pred_quant = quantizable_model(a.float().to(device), x.float().to(device), z.float().to(device), b.float().to(device))
                #y_pred_quant = quantizable_model(a.float().to(dtype).to(device), x.float().to(dtype).to(device), z.float().to(dtype).to(device), b.float().to(dtype).to(device))
                end = time.time()
                losses_quant.append(loss_fn(y_pred_quant, y.float().to(device)).item())
                execution_time_quant.append(end-start)
        
        prof.step()
ba = time.time()
print('Total Time', ba-ab)

print('Quantized test loss', sum(losses_quant)/len(losses_quant))
print('Quantized test execution time', sum(execution_time_quant[initial:])/len(execution_time_quant[initial:]))
print(execution_time_quant[initial:])

print_nvidia_smi()

