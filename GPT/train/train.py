
import pandas as pd
import argparse
from utils import set_seed
import numpy as np
# import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import math
from utils import SmilesEnumerator
import re

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    # in moses dataset, on average, there are only 5 molecules per scaffold
    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--data_name', type=str, default='moses2',
                        help="name of the dataset to train on", required=False)
    # parser.add_argument('--property', type=str, default = 'qed', help="which property to use for condition", required=False)
    parser.add_argument('--props', nargs="+", default=['qed'],
                        help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default=0, help="number of properties to use for condition",
                        required=False)
    # parser.add_argument('--prop1_unique', type=int, default = 0, help="unique values in that property", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=512,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=int,
                        default=6e-4, help="learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="number of layers in lstm", required=False)

    args = parser.parse_args()

    set_seed(42)

    data = pd.read_csv('chemprop\\preds_path\\' + 'regression_train_fingerprint.csv')

    data = data.dropna(axis=0).reset_index(drop=True)

    data.columns = data.columns.str.lower()

    train_data = data.reset_index(
        drop=True)

    data = pd.read_csv('chemprop\\preds_path\\' + 'regression_val_fingerprint.csv')
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    val_data = data.reset_index(
        drop=True)

    smiles = train_data['smiles']
    vsmiles = val_data['smiles']
    prop = list(np.array(train_data.iloc[:, 1:]))
    print('prop len: ', len(prop))
    vprop = list(np.array(val_data.iloc[:, 1:]))
    num_props = args.num_props


    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    lens = [len(regex.findall(i.strip()))
            for i in list(smiles.values)]
    max_len = 242
    print('Max len: ', max_len)

    smiles = [i + str('<') * (max_len - len(regex.findall(i.strip())))
              for i in smiles]
    vsmiles = [i + str('<') * (max_len - len(regex.findall(i.strip())))
               for i in vsmiles]


    whole_string = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B',
                    'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]',
                    '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]',
                    '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]',
                    '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]',
                    '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]',
                    '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p',
                    's',
                    '.', '[Na+]', '[Cl-]', '[K+]', '[Hg]', '[Zn]', '[Br-]', '[Sb]', '[Zn+2]', '[PbH2]', '[LiH]', '[As]',
                    '[Ca+2]', '[Pt]', '[Fe-2]', '[AsH]', '[Co]', '[Fe+2]', '[I-]', '[Bi]', '[NH4+]', '[Gd+3]', '[Ca]',
                    '[Gd]',
                    '[AlH3]', '[Li+]', '[Cl+3]',
                    '[SH-]', '[Cl+2]', '[Cl+]', '[SeH2]', '[P-]', '[F-]', '[I+2]', '[Br+2]', '[Se-]', '[I+3]']
    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop=prop, aug_prob=0)
    # train_data=torch.from_numpy(np.array(train_data)).to(device)
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop=vprop, aug_prob=0)
    # valid_dataset=torch.from_numpy(np.array(valid_data)).to(device)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props=num_props,  # args.num_props,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      lstm=args.lstm, lstm_layers=args.lstm_layers)
    model = GPT(mconf)
    # model.to(device)
    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                          lr_decay=True, warmup_tokens=0.1 * len(train_data) * max_len,
                          final_tokens=args.max_epochs * len(train_data) * max_len,
                          num_workers=10, ckpt_path=f'{args.run_name}.pt', block_size=train_dataset.max_len,
                          generate=False)
    trainer = Trainer(model, train_dataset, valid_dataset,
                      tconf, train_dataset.stoi, train_dataset.itos)
    df = trainer.train()
