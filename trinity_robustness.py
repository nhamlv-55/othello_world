import math
import time
import numpy as np
from copy import deepcopy
import pickle
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from torch.utils.data import Subset
from tqdm import tqdm

from data import get_othello, plot_probs, plot_mentals
from data.othello import permit, start_hands, OthelloBoardState, permit_reverse
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig, GPTforProbeIA
from mingpt.utils import sample, intervene, print_board
from mingpt.probe_model import BatteryProbeClassification, BatteryProbeClassificationTwoLayer

from mingpt.utils import set_seed
set_seed(44)
DEVICE = torch.device('cpu')

def get_probe()->nn.Module:
    championship = False
    mid_dim = 256
    how_many_history_step_to_use = 99
    exp = f"state_tl{mid_dim}"
    if championship:
        exp += "_championship"

    layer = 8
    probe = BatteryProbeClassificationTwoLayer(DEVICE, probe_class=3, num_task=64, mid_dim=mid_dim)
    load_res = probe.load_state_dict(torch.load(f"./ckpts/battery_othello/{exp}/layer{layer}/checkpoint.ckpt", map_location=DEVICE))
    probe.eval()
    return probe

def get_OthelloGPT(probe_layer: int)->nn.Module:


    mconf = GPTConfig(61, 59, n_layer=8, n_head=8, n_embd=512)

    model = GPTforProbeIA(mconf, probe_layer=probe_layer)#, disable_last_layer_norm = True
    load_res = model.load_state_dict(torch.load("./ckpts/gpt_synthetic.ckpt", map_location=DEVICE))

    model.eval()
    return model


PROBE = get_probe()
GPT_MODEL = get_OthelloGPT(probe_layer=8)




with open("intervention_benchmark.pkl", "rb") as input_file:
    DATASET = pickle.load(input_file)

othello = get_othello(ood_perc=0., data_root=None, wthor=False, ood_num=1)
CHARSET = CharDataset(othello)

completion = DATASET[888]['history']
partial_game = torch.tensor([CHARSET.stoi[s] for s in completion], dtype=torch.long).to(DEVICE)
partial_game = partial_game[None, :]
h = GPT_MODEL.forward_1st_stage(partial_game)
#h = query_in.unsqueeze(0).unsqueeze(0)
print(h.shape)

#head is just a weight linear 512 by 61 as seen in GPT_MODEL.py, call this weight W
#call the last row in h, embedding_input, just a vector of size 512
#Vary each value in the embedding_vector upto 0.01% of its current value. Called the result vector embedding_p
#then embedding_p*W is a vector of size 61
#check that embedding_input*W and embedding_p*W decoded to the same set of next legal move
#do this on MARABOU

out1 = GPT_MODEL.head(GPT_MODEL.ln_f(h))
print("output by running head(h):\n", out1[0][-1][:], out1.shape)

out2, _ = GPT_MODEL(partial_game)
print("output by running the model(s):\n", out2[0][-1][:], out2.shape)

reconstructed_board, _ = PROBE((h)[0][-1])
print("reconstructed board:\n", reconstructed_board.squeeze()[:10])

def probe_result_to_board(reconstructed_board):
    board = torch.argmax(reconstructed_board.squeeze(), dim = -1).reshape(64).tolist()
    return board
board = probe_result_to_board(reconstructed_board)
print(board)




def verify_probe(probe: nn.Module, case_id: int, eps: float):
    game = DATASET[case_id]


