import numpy as np
import pickle
import torch.nn as nn
from tqdm import tqdm
import torch
from typing import List, Tuple, Set, Dict, Any
from mingpt.model import GPT, GPTConfig, GPTforProbeIA
from mingpt.probe_model import BatteryProbeClassification, BatteryProbeClassificationTwoLayer
import logging
from mingpt.utils import set_seed, get_OthelloGPT, get_probe, gen_dataset, build_marabou_net
from maraboupy import MarabouCore, Marabou, MarabouUtils
import random

set_seed(44)
DEVICE = torch.device('cpu')

"""
white 0, blank 1, black 2
"""
PROBE = get_probe(device=DEVICE)
GPT_MODEL: GPTforProbeIA = get_OthelloGPT(probe_layer=8, device=DEVICE)
with open("good_games.pkl", "rb") as f:
    GOOD_GAMES = pickle.load(f)
DUMMY_INPUT = GOOD_GAMES[0]['h'][0][-1] #h shoud be of the shape B * T * 512
P_ZERO = 10e-6
N_ZERO = -10e-6

MAX_TIME = 600  # in seconds

M_OPTIONS: MarabouCore.Options = Marabou.createOptions(verbosity=0,
                                                       initialSplits=4,
                                                       timeoutInSeconds=MAX_TIME,
                                                       snc=True,
                                                       numWorkers=32,
                                                       )




def verify_probe(eps: float):
    #reuse the m_probe. only input and output bounds are changed
    m_probe = build_marabou_net(PROBE, DUMMY_INPUT)

    input_vars = m_probe.inputVars[0]
    output_vars = m_probe.outputVars[0][0]

    for g in GOOD_GAMES[:1]:
        #get 4 random target tiles to flip
        targets = random.sample(range(64), 4)

        for target in targets:
            h_input = g['h'][0][-1].detach().numpy() #h is of the shape B * T * 512. B should always be 1
            logging.debug(f"h_input shape: {h_input.shape}")
            logging.debug(f"Input vars: {m_probe.inputVars[0]}")
            logging.debug(f"Output vars: {m_probe.outputVars[0]}")
            


            #set input perturbation
            assert h_input.shape == m_probe.inputVars[0].shape

            for i in range(len(h_input)): #should be 0->511
                m_probe.setLowerBound(input_vars[i], h_input[i] - eps)
                m_probe.setUpperBound(input_vars[i], h_input[i] + eps)

            #set output conditions
            #Can we flip tile Target?
            print(m_probe.outputVars[0].shape)
            print(f"Current value for tile {target}: {output_vars[target]} ~ {len(g['true_board'])}")
            print(f"True argmax of the target tile {g['true_board'][target]}")

            #Can we force target to be smaller than target + 1 % 3
            true_argmax = int(g['true_board'][target])
            target_idx = output_vars[target][true_argmax]
            adv_idx = output_vars[target][(true_argmax+1)%3]
            print(f"target_idx: {target_idx}")
            constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
            constraint.setScalar(P_ZERO)
            constraint.addAddend(-1, target_idx)
            constraint.addAddend(1, adv_idx)

            query = m_probe.getMarabouQuery()

            query.addEquation(constraint.toCoreEquation())

            query_name =  f"queries/probe_robust/game_{g['game_idx']}_target_{target}_eps_{eps}.txt"
            MarabouCore.saveQuery(query, query_name)



if __name__=="__main__":
    verify_probe(0.05)