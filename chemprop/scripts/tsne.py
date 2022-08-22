import os
import sys
import time
from typing import List

from matplotlib import offsetbox
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.manifold import TSNE
from tap import Tap
from tqdm import tqdm
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import get_smiles
from chemprop.features import get_features_generator
from chemprop.utils import makedirs


class Args(Tap):
    smiles_paths: List[str]  # Path to .csv files containing smiles strings (with header)
    smiles_column: str = None  # Name of the column containing SMILES strings for the first data. By default, uses the first column.
    colors: List[str] = ['red',  'green','purple','orange',  'blue']  # Colors of the points associated with each dataset
    sizes: List[float] = [1, 1, 1, 1, 1]  # Sizes of the points associated with each molecule
    scale: int = 1  # Scale of figure
    plot_molecules: bool = False  # Whether to plot images of molecules instead of points
    max_per_dataset: int = 10000  # Maximum number of molecules per dataset; larger datasets will be subsampled to this size
    save_path: str  # Path to a .png file where the t-SNE plot will be saved
    cluster: bool = False  # Whether to create new clusters from all smiles, ignoring original csv groupings


def compare_datasets_tsne(args: Args):
    if len(args.smiles_paths) > len(args.colors) or len(args.smiles_paths) > len(args.sizes):
        raise ValueError('Must have at least as many colors and sizes as datasets')

    # Random seed for random subsampling
    np.random.seed(0)

    # Load the smiles datasets
    print('Loading data')
    smiles, slices, labels = [], [], []
    label=0
    for smiles_path in args.smiles_paths:
        # Get label
        label = label+1

        # Get SMILES
        new_smiles = get_smiles(path=smiles_path, smiles_columns=args.smiles_column, flatten=True)
        print(f'{label}: {len(new_smiles):,}')

        # Subsample if dataset is too large
        if len(new_smiles) > args.max_per_dataset:
            print(f'Subsampling to {args.max_per_dataset:,} molecules')
            new_smiles = np.random.choice(new_smiles, size=args.max_per_dataset, replace=False).tolist()

        slices.append(slice(len(smiles), len(smiles) + len(new_smiles)))
        labels.append(label)
        smiles += new_smiles

    # Compute Morgan fingerprints
    print('Computing Morgan fingerprints')
    morgan_generator = get_features_generator('morgan')
    morgans = [morgan_generator(smile) for smile in tqdm(smiles, total=len(smiles))]
    # Compute latent fingerprints
    filename = 'E:\\研一\\code\\chemprop-master\\preds_path\\regression_fingerprint.csv'
    data = pd.read_csv(filename, sep=',')
    latent = [list(np.array(data.iloc[i, 1:])) for i in range(2335)]
    #label=label+1
    #labels=[label for i in range(2335)]


    print('Running t-SNE')
    start = time.time()
    tsne = TSNE(n_components=2, init='pca', random_state=0, metric='jaccard')
    X = tsne.fit_transform(morgans)
    Y= tsne.fit_transform(latent)
    print(f'time = {time.time() - start:.2f} seconds')

    if args.cluster:
        import hdbscan  # pip install hdbscan
        print('Running HDBSCAN')
        start = time.time()
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
        colors = clusterer.fit_predict(X)
        print(f'time = {time.time() - start:.2f} seconds')

    print('Plotting t-SNE')
    '''x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    y_min, y_max = np.min(Y, axis=0), np.max(Y, axis=0)
    Y = (Y - y_min) / (y_max - y_min)'''

    makedirs(args.save_path, isfile=True)

    plt.clf()
    fontsize = 50 * args.scale
    fig = plt.figure(figsize=(64 * args.scale, 48 * args.scale))
    plt.title('t-SNE using Morgan fingerprint and latent vector with Jaccard similarity', fontsize=2 * fontsize)
    ax = fig.gca()
    handles = []
    legend_kwargs = dict(loc='upper right', fontsize=fontsize)

    if args.cluster:
        plt.scatter(X[:, 0], X[:, 1], s=150 * np.mean(args.sizes), c=colors, cmap='nipy_spectral')
    else:
        for slc, color, label, size in zip(slices, args.colors, labels, args.sizes):
            if args.plot_molecules:
                # Plots molecules
                handles.append(mpatches.Patch(color=color, label=label))

                for smile, (x, y) in zip(smiles[slc], X[slc]):
                    img = Draw.MolsToGridImage([Chem.MolFromSmiles(smile)], molsPerRow=1, subImgSize=(200, 200))
                    imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img), (x, y), bboxprops=dict(color=color))
                    ax.add_artist(imagebox)
            else:
                # Plots points
                #plt.scatter(X[slc, 0], X[slc, 1], s=150 * size, color=color, label=label)
                X1=X[0:120]
                X2=X[120:]
                plt.scatter(X1[slc, 0], X1[slc, 1], s=150 * size, color='red', label=label)
                plt.scatter(X2[slc, 0], X2[slc, 1], s=150 * size, color='green', label=label)


        if args.plot_molecules:
            legend_kwargs['handles'] = handles

    plt.sca(ax)
    label=label+1
    #plt.scatter(Y[slc, 0], Y[slc, 1], s=150 * size, color='blue', label=label)
    Y1 = Y[0:120]
    Y2 = Y[120:]
    plt.scatter(Y1[slc, 0], Y1[slc, 1], s=150 * size, color='blue', label=label)
    plt.scatter(Y2[slc, 0], Y2[slc, 1], s=150 * size, color='purple', label=label)

    #plt.legend(**legend_kwargs)
    plt.legend(['Morgan_p','Morgan_n','latent_p','latent_n'],loc='upper right', fontsize=fontsize)
    plt.xticks([]), plt.yticks([])

    print('Saving t-SNE')
    plt.savefig(args.save_path)

    '''
    plt.figure()
    from scipy.interpolate import Rbf  # 引入径向基函数

    df1 = Y[slc,0]  # 读取第一列数据
    df2 = Y[slc,1]  # 读取第二列数据
    f = pd.read_csv('E:\\研一\\code\\chemprop-master\\data\\train_regression.csv')
    df3 = f['Mean_Inhibition'] # 读取第三列数据

    odf1 = np.linspace(min(Y[slc,0]), max(Y[slc,0]), 50)  # 设置网格经度
    odf2 = np.linspace(min(Y[slc,1]), max(Y[slc,1]), 50)  # 设置网格纬度
    odf1, odf2 = np.meshgrid(odf1, odf2)  # 网格化
    func = Rbf(df1, df2, df3, function='linear')  # 定义插值函数plt.cm.hot
    odf3_new = func(odf1, odf2)  # 获得插值后的网格累计降水量
    plt.contourf(odf1, odf2, odf3_new,
                 levels=np.arange(odf3_new.min(), odf3_new.max(), (odf3_new.max() - odf3_new.min()) / 10), cmap='GnBu',
                 extend='both')  # 画图
    # 添加等高线
    #line = plt.contour(odf1, odf2, odf3_new,evels=np.arange(odf3_new.min(), odf3_new.max(), (odf3_new.max() - odf3_new.min()) / 10))
    #plt.clabel(line, inline=True, fontsize=12)
    plt.title('Latent space')
    plt.axis('off')
    plt.colorbar()
    #plt.clim(0, 3)
    plt.savefig('latent.png')
    plt.show()

    '''


if __name__ == '__main__':
    compare_datasets_tsne(Args().parse_args())
