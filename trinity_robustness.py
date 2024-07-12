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
from typing import List, Tuple, Set
from data import get_othello, plot_probs, plot_mentals
from data.othello import permit, start_hands, OthelloBoardState, permit_reverse
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig, GPTforProbeIA
from mingpt.utils import sample, intervene, print_board
from mingpt.probe_model import BatteryProbeClassification, BatteryProbeClassificationTwoLayer

from mingpt.utils import set_seed, get_OthelloGPT, get_probe, gen_dataset
set_seed(44)
DEVICE = torch.device('cpu')


PROBE = get_probe(device=DEVICE)
GPT_MODEL = get_OthelloGPT(probe_layer=8, device=DEVICE)
GOOD_GAMES = gen_dataset(device=DEVICE, save_location='good_games.pkl')
