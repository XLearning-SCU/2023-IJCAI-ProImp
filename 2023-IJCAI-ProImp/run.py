# =====================
# Incomplete Multi-view Clustering via Prototype-based Imputation
# =====================
# Author: Haobin Li
# Date: July, 2023
# E-mail: haobinli.gm@gmail.com,

# @article{li2023incomplete,
#  title={Incomplete Multi-view Clustering via Prototype-based #Imputation},
#   author={Li, Haobin and Li, Yunfan and Yang, Mouxing and Hu, Peng #and Peng, Dezhong and Peng, Xi},
#   journal={arXiv preprint arXiv:2301.11045},
#   year={2023}
# }
# =====================

import argparse
import collections
import itertools
import torch
from model import ProImp
from get_mask import get_mask
from util import cal_std, get_logger
from datasets import *
from configure import get_default_config

dataset = {
    0: "Scene_15",
    1: "NoisyMNIST",
    2: "Reuters",
    3: "MNIST-USPS",
    4: "cub_googlenet",
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='0', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='50',
                    help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='5',
                    help='number of test times')

args = parser.parse_args()
dataset = dataset[args.dataset]


def main():
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset)
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    logger = get_logger()

    logger.info('Dataset:' + str(dataset))
    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("          %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))

    # Load data
    X_list, Y_list = load_data(config)
    x1_train_raw = X_list[0]
    x2_train_raw = X_list[1]
    accumulated_metrics = collections.defaultdict(list)

    for data_seed in range(1, args.test_time + 1):
        # Get the Mask
        np.random.seed(data_seed)
        mask = get_mask(
            2, x1_train_raw.shape[0], config['training']['missing_rate'])
        # mask the data
        x1_train = x1_train_raw * mask[:, 0][:, np.newaxis]
        x2_train = x2_train_raw * mask[:, 1][:, np.newaxis]

        x1_train = torch.from_numpy(x1_train).float().to(device)
        x2_train = torch.from_numpy(x2_train).float().to(device)
        x1_train = (x1_train-x1_train.mean())/(x1_train.std())
        x2_train = (x2_train-x2_train.mean())/(x2_train.std())
        mask = torch.from_numpy(mask).long().to(device)

        # Set random seeds
        if config['training']['missing_rate'] == 0:
            seed = data_seed
        else:
            seed = config['training']['seed']
        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Build the model
        PROIMP = ProImp(config)
        optimizer = torch.optim.Adam(
            itertools.chain(PROIMP.autoencoder1.parameters(), PROIMP.autoencoder2.parameters(),
                            PROIMP.dual_attention.parameters()),
            lr=config['training']['lr']
        )
        PROIMP.to_device(device)

        # Print the models
        logger.info(PROIMP.autoencoder1)
        logger.info(PROIMP.dual_attention)
        logger.info(optimizer)

        # Training
        acc, nmi, ari = PROIMP.train(config, logger, x1_train, x2_train, Y_list,
                                     mask, optimizer, device)
        accumulated_metrics['acc'].append(acc)
        accumulated_metrics['nmi'].append(nmi)
        accumulated_metrics['ari'].append(ari)

    logger.info('--------------------Training over--------------------')
    cal_std(logger, accumulated_metrics['acc'],
            accumulated_metrics['nmi'], accumulated_metrics['ari'])


if __name__ == '__main__':
    main()
