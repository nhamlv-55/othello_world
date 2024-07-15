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
from enum import Enum
set_seed(44)
DEVICE = torch.device('cpu')

"""
white 0, blank 1, black 2
"""
PROBE = get_probe(device=DEVICE)
GPT_MODEL: GPTforProbeIA = get_OthelloGPT(probe_layer=8, device=DEVICE)

# GOOD_GAMES = gen_dataset(DEVICE, 'good_games_after_layernom.pkl')
with open("good_games_after_layernom.pkl", "rb") as f:
    GOOD_GAMES = pickle.load(f)
DUMMY_INPUT = GOOD_GAMES[0]['h'][0][-1] #h shoud be of the shape B * T * 512
P_ZERO = 10e-6
N_ZERO = -10e-6

MAX_TIME = 600  # in seconds

M_OPTIONS: MarabouCore.Options = Marabou.createOptions(verbosity=0,
                                                       initialSplits=4,
                                                       timeoutInSeconds=MAX_TIME,
                                                       snc=False,
                                                       numWorkers=32,
                                                       )

M_OPTIONS: MarabouCore.Options = Marabou.createOptions()

class TrinityProperty(Enum):
    PROBE_ROBUST = 1
    HEAD_ROBUST = 2
    PROBE_PROVE_HEAD = 3
    HEAD_PROVE_PROBE = 4

def gen_verify_queries(eps: float, n_games: int, n_targets: int, task: TrinityProperty ):
    #reuse the m_probe. only input and output bounds are changed
    m_probe = build_marabou_net(PROBE, DUMMY_INPUT)

    m_head = build_marabou_net(GPT_MODEL.head, DUMMY_INPUT)



    for g in tqdm(GOOD_GAMES[:n_games]):
        input_vars = m_probe.inputVars[0]
        output_vars = m_probe.outputVars[0][0]
        #set output conditions
        if task==TrinityProperty.PROBE_ROBUST:
            """
            generate the constraints for probe robustness
            """
            #get 4 random target tiles to flip
            targets = random.sample(range(64), n_targets)


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

            query_name =  f"queries/probe_robust/group_{g['game_idx']%8}/game_{g['game_idx']}_target_{target}_eps_{eps}.txt"
            MarabouCore.saveQuery(query, query_name)

        elif task==TrinityProperty.HEAD_ROBUST:
            """
            generate the constrainst for head robustness
            """
            input_vars = m_head.inputVars[0]
            output_vars = m_head.outputVars[0]
            print("output vars", output_vars)
            true_prediction = np.array(g['true_output'])
            true_prediction = true_prediction > 0
            true_prediction = true_prediction.astype(int).tolist()
            # true_prediction = true_prediction[1:30] + [-1, -1, -1, -1] + true_prediction[30:]
            true_prediction = np.array(true_prediction[1:] + [-1, -1, -1, -1]).reshape(8,8)
            print(true_prediction)
            true_pred_label = np.argmax(true_prediction)

            print("true pred label", true_pred_label)
            print("true valid moves", g['true_valid_moves'])
            adv_labels = set(range(1, 61))
            adv_labels.remove(true_pred_label)
            adv_labels = sorted(adv_labels)
            adv_labels = random.sample(adv_labels, n_targets)
            print("adv_labels:", adv_labels)
            h_input = g['h'][0][-1].detach().numpy() #h is of the shape B * T * 512. B should always be 1

            #set input perturbation
            assert h_input.shape == m_head.inputVars[0].shape

            for i in range(len(h_input)): #should be 0->511
                m_head.setLowerBound(input_vars[i], h_input[i] - eps)
                m_head.setUpperBound(input_vars[i], h_input[i] + eps)

            for adv_l in adv_labels:
                #set output constraint. Can we make some random move to be the best move?
                query_name = f"queries/head_robust/group_{g['game_idx']%8}/game_{g['game_idx']}_adv_labels_{adv_l}_eps_{eps}.txt"
                query = m_head.getMarabouQuery()
                
                constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
                constraint.setScalar(P_ZERO)
                constraint.addAddend(-1, output_vars[true_pred_label])
                constraint.addAddend(1, output_vars[adv_l])
               
                query.addEquation(constraint.toCoreEquation())

                MarabouCore.saveQuery(query, query_name)

if __name__=="__main__":
    # gen_verify_probe_queries(eps = 0.1, n_games= 100, n_targets=4)
    gen_verify_queries(eps = 0.9, n_games=1, n_targets=4, task=TrinityProperty.HEAD_ROBUST)