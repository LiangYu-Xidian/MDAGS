import sys

sys.path.append('moses')
import random
import pandas as pd
from utils import check_novelty, sample, canonic_smiles
from dataset import SmileDataset
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
import math
from tqdm import tqdm
import argparse
from model import GPT, GPTConfig

import torch
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from moses.utils import get_mol
import re
import moses
import json
from rdkit.Chem import RDConfig
# import json

import os
import sys

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
    parser.add_argument('--scaffold', action='store_true', default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format", required=True)
    parser.add_argument('--data_name', type=str, default='moses2', help="name of the dataset to train on",
                        required=False)
    parser.add_argument('--batch_size', type=int, default=512, help="batch size", required=False)
    parser.add_argument('--gen_size', type=int, default=10000, help="number of times to generate from a batch",
                        required=False)
    parser.add_argument('--vocab_size', type=int, default=131, help="number of layers",
                        required=False)  # previously 28 .... 26 for moses. 94 for guacamol
    parser.add_argument('--block_size', type=int, default=242, help="number of layers",
                        required=False)  # previously 57... 54 for moses. 100 for guacamol.
    parser.add_argument('--props', nargs="+", default=['fingerprint'], help="properties to be used for condition",
                        required=False)
    parser.add_argument('--n_layer', type=int, default=8, help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8, help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256, help="embedding dimension", required=False)
    parser.add_argument('--lstm_layers', type=int, default=2, help="number of layers in lstm", required=False)

    args = parser.parse_args()

    context = "C"

    seed = 2
    random.seed(seed)
    data = pd.read_csv('GPT\\datasets\\' + 'guacamol_train.smiles')
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    data = data.reset_index(
        drop=True)
    tsmiles = data['smiles']
    data = pd.read_csv('chemprop\\preds_path\\seed\\' + 'seed_fingerprint_GP.csv')
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    data = data.reset_index(
        drop=True)  # 'split' instead of 'source' in moses
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    whole_string = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=',
                    'B',
                    'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]',
                    '[C+]',
                    '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]',
                    '[N+]',
                    '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]',
                    '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]',
                    '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]',
                    '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o',
                    'p',
                    's',
                    '.', '[Na+]', '[Cl-]', '[K+]', '[Hg]', '[Zn]', '[Br-]', '[Sb]', '[Zn+2]', '[PbH2]', '[LiH]',
                    '[As]',
                    '[Ca+2]', '[Pt]', '[Fe-2]', '[AsH]', '[Co]', '[Fe+2]', '[I-]', '[Bi]', '[NH4+]', '[Gd+3]',
                    '[Ca]', '[Gd]',
                    '[AlH3]', '[Li+]', '[Cl+3]',
                    '[SH-]', '[Cl+2]', '[Cl]', '[SeH2]', '[P-]', '[F-]', '[I+2]', '[Br+2]', '[Se-]', '[I+3]']
    chars = sorted(list(set(whole_string)))
    stoi = {ch: i for i, ch in enumerate(chars)}

    itos = {i: ch for ch, i in stoi.items()}

    print(itos)
    print(len(itos))

    num_props = len(args.props)
    mconf = GPTConfig(args.vocab_size, args.block_size, num_props=num_props,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      lstm=args.lstm, lstm_layers=args.lstm_layers)

    model = GPT(mconf)

    weight = torch.load(args.model_weight)
    model.load_state_dict(weight, strict=False)
    print(model)
    model.to('cuda')
    print('Model loaded')

    prop_condition = data.iloc[2:, :]

    all_dfs = []
    all_metrics = []

    count = 0
    gen_iter = math.ceil(len(prop_condition) / args.batch_size)
    if (prop_condition is not None):
        count = 0
        for i in range(gen_iter):
            start = i * args.batch_size
            end = (i + 1) * args.batch_size if (i + 1) * args.batch_size < len(prop_condition) else len(prop_condition)
            c = list(np.array(prop_condition.iloc[start:end, :]))

            molecules = []
            count += 1
            print('count', count)

            x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...].repeat(end - start,
                                                                                                            1).to(
                'cuda')
            print('x_size', x.shape)
            p = None
            if len(args.props) == 1:
                p = torch.tensor(c).unsqueeze(1).to(torch.float32).to(
                    'cuda')
            else:
                p = torch.tensor(c).to(torch.float32).to(
                    'cuda')
            sca = None
            print('p', p)
            print('p_size', p.shape)
            y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop=p,
                       scaffold=sca)
            for gen_mol in y:
                completion = ''.join([itos[int(i)] for i in gen_mol])
                completion = completion.replace('<', '')
                mol = get_mol(completion)
                if mol:
                    molecules.append(mol)

            "Valid molecules % = {}".format(len(molecules))

            mol_dict = []

            for i in molecules:
                mol_dict.append({'molecule': i, 'smiles': Chem.MolToSmiles(i)})

            if len(mol_dict) == 0:
                continue
            results = pd.DataFrame(mol_dict)

            all_dfs.append(results)
    print('count', end)
    results = pd.concat(all_dfs)
    results.to_csv(args.csv_name + '.csv', index=False)

    unique_smiles = list(set(results['smiles']))
    canon_smiles = [canonic_smiles(s) for s in results['smiles']]
    unique_smiles = list(set(canon_smiles))
    novel_ratio = check_novelty(unique_smiles, set(tsmiles))

    print('Valid ratio: ', np.round(len(results) / len(prop_condition), 4))
    print('Unique ratio: ', np.round(len(unique_smiles) / len(results), 4))
    print('Novelty ratio: ', np.round(novel_ratio / 100, 4))

