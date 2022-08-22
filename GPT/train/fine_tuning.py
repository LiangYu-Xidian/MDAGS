
import sys
import pandas
from dataset import SmileDataset
sys.path.append('moses')

import re
from rdkit.Chem import RDConfig
import json
import pandas as pd
import argparse
#from utils import set_seed
import numpy as np
import torch

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str,help="name for run", required=False)
    parser.add_argument('--debug', action='store_true',default=False, help='debug')
    parser.add_argument('--learning_rate', type=int, default=6e-4, help="learning rate", required=False)
    parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
    parser.add_argument('--scaffold', action='store_true', default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format", required=False)
    parser.add_argument('--data_name', type=str, default='moses2', help="name of the dataset to train on",
                        required=False)
    parser.add_argument('--batch_size', type=int, default=512, help="batch size", required=False)
    parser.add_argument('--gen_size', type=int, default=10000, help="number of times to generate from a batch",
                        required=False)
    parser.add_argument('--vocab_size', type=int, default=131, help="number of layers",
                        required=False)  # previously 28 .... 26 for moses. 94 for guacamol
    parser.add_argument('--block_size', type=int, default=242, help="number of layers",
                        required=False)  # previously 57... 54 for moses. 100 for guacamol.
    parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)    #我加的
    parser.add_argument('--props', nargs="+", default=['fingerprint'], help="properties to be used for condition",
                        required=False)
    parser.add_argument('--n_layer', type=int, default=8, help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8, help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256, help="embedding dimension", required=False)
    parser.add_argument('--lstm_layers', type=int, default=2, help="number of layers in lstm", required=False)
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="total epochs", required=False)

    args = parser.parse_args()

    context = "C"

    data = pd.read_csv('chemprop\\preds_path\\' + 'regression_train_fingerprint.csv')
    # data = pd.read_csv('datasets/' + args.data_name + '.csv')
    data = data.dropna(axis=0).reset_index(drop=True)
    # data = data.sample(frac = 0.1).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    train_data = data.reset_index(
        drop=True)  # 'split' instead of 'source' in moses
    smiles = train_data['smiles']
    data = pd.read_csv('chemprop\\preds_path\\' + 'regression_val_fingerprint.csv')
    # data = pd.read_csv('datasets/' + args.data_name + '.csv')
    data = data.dropna(axis=0).reset_index(drop=True)
    # data = data.sample(frac = 0.1).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    val_data = data.reset_index(
        drop=True)  # 'split' instead of 'source' in moses
    vsmiles = val_data['smiles']

    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    max_len=242

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
                     '[SH-]','[Cl+2]','[Cl+]','[SeH2]','[P-]','[F-]','[I+2]','[Br+2]','[Se-]','[I+3]']
    chars = sorted(list(set(whole_string)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    # stoi = json.load(open(f'{args.data_name}_stoi.json', 'r'))

    # itos = { i:ch for i,ch in enumerate(chars) }
    itos = {i: ch for ch, i in stoi.items()}

    print(itos)
    print(len(itos))

    prop = list(np.array(train_data.iloc[:, 1:]))
    vprop = list(np.array(val_data.iloc[:, 1:]))
    num_props = 1
    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop=prop, aug_prob=0)
    val_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0)
    mconf = GPTConfig(args.vocab_size, args.block_size, num_props=num_props,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      lstm=args.lstm, lstm_layers=args.lstm_layers)

    model = GPT(mconf)
    # summary(model=model, input_size=(), device="cpu")

    weight = torch.load(args.model_weight)
    model.load_state_dict(weight, strict=False)
    print('Model loaded')
    print(model)

    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                          lr_decay=True, warmup_tokens=0.1 * len(train_data) * max_len,
                          final_tokens=args.max_epochs * len(train_data) * max_len,
                          num_workers=10, ckpt_path=f'{args.run_name}.pt', block_size=train_dataset.max_len,
                          generate=False)
    trainer = Trainer(model, train_dataset, val_dataset,
                      tconf, train_dataset.stoi, train_dataset.itos)
    df = trainer.train()
