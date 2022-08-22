import torch  # 命令行是逐行立即执行的
from collections import OrderedDict
import csv
from typing import List, Optional, Union, Tuple

import numpy as np
from tqdm import tqdm

#from .predict import predict
from chemprop.spectra_utils import normalize_spectra, roundrobin_sid
from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.utils import load_args, load_checkpoint, load_scalers, makedirs, timeit, update_prediction_args
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim, set_reaction, set_explicit_h, set_adding_hs, reset_featurization_parameters
from chemprop.models import MoleculeModel
data_path=''


def load_data(data_path):
    """
    Function to load data from a list of smiles or a file.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: A list of list of smiles, or None if data is to be read from file
    :return: A tuple of a :class:`~chemprop.data.MoleculeDataset` containing all datapoints, a :class:`~chemprop.data.MoleculeDataset` containing only valid datapoints,
                 a :class:`~chemprop.data.MoleculeDataLoader` and a dictionary mapping full to valid indices.
    """
    print('Loading data')

    full_data = get_data(path=data_path.test_path, smiles_columns=data_path.smiles_columns, target_columns=[], ignore_columns=[],
                             skip_invalid_smiles=False, args=data_path, store_row=not data_path.drop_extra_columns)

    print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

    print(f'Test size = {len(test_data):,}')

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data
    )

    return full_data, test_data, test_data_loader, full_to_valid_indices


content = torch.load('t.pth')
print(content.keys())   # keys()
# 之后有其他需求比如要看 key 为 model 的内容有啥
print(content['model'])
full_data, test_data, test_data_loader, full_to_valid_indices = load_data(data_path)
for batch in tqdm(test_data_loader, leave=False):
    # Prepare batch
    batch: MoleculeDataset
    mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch = \
        batch.batch_graph(), batch.features(), batch.atom_descriptors(), batch.atom_features(), batch.bond_features()

    # Make predictions
    with torch.no_grad():
        batch_preds = content(mol_batch, features_batch, atom_descriptors_batch,
                            atom_features_batch, bond_features_batch)

    preds.extend(batch_preds)

