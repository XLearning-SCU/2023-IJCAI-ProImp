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
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from munkres import Munkres
import evaluation
from util import next_batch
from torch import Tensor, nn
import math
from sklearn import metrics
import numpy as np


class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' %
                                     self._activation)
                if i < self._dim - 2:
                    encoder_layers.append(nn.Dropout(0.2))

        self._encoder = nn.Sequential(*encoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
        """
        latent = self.encoder(x)

        return latent


class ProImp():
    """ProImp module."""

    def __init__(self,
                 config):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        # super(ProImp, self).__init__()
        self._config = config

        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        # View-specific autoencoders
        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'])

        self.dim = config['training']['dim']
        self.dual_attention = dual_attention(
            dim=self.dim, num=config['training']['num']).cuda()

    def to_device(self, device):
        """ to cuda if gpu is used """
        self.autoencoder1.to(device)
        self.autoencoder2.to(device)

    def train(self, config, logger, x1_train, x2_train, Y_list, mask, optimizer, device):
        """Training the model.

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari

        """

        # Get complete data for training
        flag = (torch.LongTensor([1, 1]).to(device) == mask).int()
        flag = (flag[:, 1] + flag[:, 0]) == 2
        train_view1 = x1_train[flag]
        train_view2 = x2_train[flag]
        index_swap = None
        for epoch in range(config['training']['epoch']):
            X1, X2 = shuffle(train_view1, train_view2)
            loss_all, loss_ins, loss_pro = 0, 0, 0
            for batch_x1, batch_x2, batch_No in next_batch(X1, X2, config['training']['batch_size']):
                z_1 = self.autoencoder1.encoder(batch_x1)
                z_2 = self.autoencoder2.encoder(batch_x2)

                if epoch < config['training']['pretrain_epoch']:
                    ins_loss, pro_loss = self.dual_attention(z_1, z_2, index_swap)
                    loss = ins_loss
                else:
                    if epoch == config['training']['pretrain_epoch'] and index_swap is None:
                        index_swap = self.get_index(
                            config, mask, x1_train, x2_train)
                    ins_loss, pro_loss = self.dual_attention(z_1, z_2, index_swap)
                    loss = ins_loss + pro_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_all += loss.item()
                loss_ins += ins_loss.item()
                loss_pro += pro_loss.item()

            if (epoch + 1) % config['print_num'] == 0:
                output = "Epoch : {:.0f}/{:.0f} ===> Learing Rate = {:.4f}" \
                         "===> Ins loss = {:.4e} ===> Clu loss = {:.4e} ===> Loss = {:.4e}" \
                    .format((epoch + 1), config['training']['epoch'], optimizer.state_dict()['param_groups'][0]['lr'],
                            loss_ins, loss_pro, loss_all)

                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (epoch + 1) % config['print_num'] == 0:
                scores = self.evaluation(
                    config, logger, mask, x1_train, x2_train, Y_list, device, epoch, index_swap)

        return scores['kmeans']['accuracy'], scores['kmeans']['NMI'], scores['kmeans']['ARI']

    def evaluation(self, config, logger, mask, x1_train, x2_train, Y_list, device, epoch, swap):
        with torch.no_grad():
            self.autoencoder1.eval(), self.autoencoder2.eval(), self.dual_attention.eval()
            img_idx_eval = mask[:, 0] == 1
            txt_idx_eval = mask[:, 1] == 1
            img_missing_idx_eval = mask[:, 0] == 0
            txt_missing_idx_eval = mask[:, 1] == 0

            imgs_latent_eval_old = self.autoencoder1.encoder(
                x1_train[img_idx_eval])
            txts_latent_eval_old = self.autoencoder2.encoder(
                x2_train[txt_idx_eval])

            common_idx_eval = (mask[:, 0] == 1) & (mask[:, 1] == 1)
            imgs_latent_eval_common = self.autoencoder1.encoder(
                x1_train[common_idx_eval])
            txts_latent_eval_common = self.autoencoder2.encoder(
                x2_train[common_idx_eval])
            if epoch < config['training']['pretrain_epoch']:
                index_swap = self.dual_attention.get_swap(
                    imgs_latent_eval_common, txts_latent_eval_common, config['training']['num'])
            else:
                index_swap = swap

            imgs_latent_eval, txts_latent_eval, _, _, _, _ = self.dual_attention.eval_z(imgs_latent_eval_old,
                                                                                        txts_latent_eval_old, index_swap, epoch)

            # representations
            latent_code_img_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(
                device)
            latent_code_txt_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(
                device)

            if x2_train[img_missing_idx_eval].shape[0] != 0:
                img_missing_latent_eval = self.autoencoder2.encoder(
                    x2_train[img_missing_idx_eval])  # txt
                txt_missing_latent_eval = self.autoencoder1.encoder(
                    x1_train[txt_missing_idx_eval])  # img

                img2txt_recon_eval, txt2img_recon_eval = self.dual_attention.dual_pre(txt_missing_latent_eval,
                                                                                      img_missing_latent_eval, index_swap)
                latent_code_img_eval[img_missing_idx_eval] = txt2img_recon_eval
                latent_code_txt_eval[txt_missing_idx_eval] = img2txt_recon_eval

            latent_code_img_eval[img_idx_eval] = imgs_latent_eval
            latent_code_txt_eval[txt_idx_eval] = txts_latent_eval

            latent_fusion = torch.cat(
                [latent_code_img_eval, latent_code_txt_eval], dim=1).cpu().numpy()
            print(metrics.silhouette_score(latent_fusion, Y_list[0]))
            scores = evaluation.clustering([latent_fusion], Y_list[0])
            logger.info("\033[2;29m" + 'view_concat ' +
                        str(scores) + "\033[0m")

            self.autoencoder1.train(), self.autoencoder2.train(),
            self.dual_attention.train()
        return scores

    def get_index(self, config, mask, x1_train, x2_train):
        with torch.no_grad():
            self.autoencoder1.eval(), self.autoencoder2.eval(), self.dual_attention.eval()

            common_idx_eval = (mask[:, 0] == 1) & (mask[:, 1] == 1)
            imgs_latent_eval_common = self.autoencoder1.encoder(
                x1_train[common_idx_eval])
            txts_latent_eval_common = self.autoencoder2.encoder(
                x2_train[common_idx_eval])
            index_swap = self.dual_attention.get_swap(
                imgs_latent_eval_common, txts_latent_eval_common, config['training']['num'])
            # print(index_swap)
            self.autoencoder1.train(), self.autoencoder2.train(),
            self.dual_attention.train()
        return index_swap


class InstanceLoss(nn.Module):
    def __init__(self, batch_size=256, temperature=1):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        self.batch_size = z_i.size(0)
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class PrototypeLoss(nn.Module):
    def __init__(self, batch_size=256, temperature=1):
        super(PrototypeLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.lamda = 0.75 / self.temperature

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        self.batch_size = z_i.size(0)
        self.mask = self.mask_correlated_samples(self.batch_size // 2)
        N = self.batch_size
        sim = torch.matmul(z_i, z_j.T) / self.temperature
        sim_i_i = torch.diag(sim)
        sim_i_i = self.lamda - torch.abs(
            sim_i_i - self.lamda)
        positive_samples = (sim_i_i).reshape(N, 1)

        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class dual_attention(nn.Module):
    def __init__(self, batch_size=256, num=21, dim=128):
        super(dual_attention, self).__init__()
        self.num = num
        self.dim = dim
        self.batch_size = batch_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.l2_norm = F.normalize
        self.soft = nn.Softmax(dim=-1)
        self.ins = InstanceLoss(temperature=0.5)
        self.pro = PrototypeLoss(batch_size=num, temperature=2)
        self.num_heads = 1
        self.cross_img = cross_attn(dim=dim, num=num, head=self.num_heads)
        self.cross_txt = cross_attn(dim=dim, num=num, head=self.num_heads)
        self.proj_c = nn.Identity()
        self.proj_z = nn.Linear(dim, dim)
        self.prototype_token_1 = nn.Linear(dim, num, bias=False)
        self.prototype_img = self.prototype_token_1.weight
        self.prototype_token_2 = nn.Linear(dim, num, bias=False)
        self.prototype_txt = self.prototype_token_2.weight
        self.projector_rep = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(dim, dim)
        )
        self.projector_prototype = nn.Identity()
        self.projector1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.projector2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def pre(self, z_img, prototype_img, z_txt, prototype_txt):
        prototype_img = F.normalize(prototype_img, dim=1)
        prototype_txt = F.normalize(prototype_txt, dim=1)
        z_img = F.normalize(z_img, dim=1)
        z_txt = F.normalize(z_txt, dim=1)
        prototype_img = self.projector_prototype(prototype_img)
        prototype_txt = self.projector_prototype(prototype_txt)
        z_img = self.projector_rep(z_img)
        z_txt = self.projector_rep(z_txt)
        return z_img, prototype_img, z_txt, prototype_txt

    def get_swap(self, z_img, z_txt, num):
        z_img_pre, prototype_img, z_txt_pre, prototype_txt = self.pre(
            z_img, self.prototype_img, z_txt, self.prototype_txt)
        _, _, attn_img, v_img, _ = self.cross_img(z_img_pre, prototype_img)
        _, _, attn_txt, v_txt, _ = self.cross_txt(z_txt_pre, prototype_txt)
        label_img = torch.argmax(attn_img.squeeze(
            0), dim=1).detach().cpu().numpy()
        label_txt = torch.argmax(attn_txt.squeeze(
            0), dim=1).detach().cpu().numpy()
        confusion_matrix = metrics.confusion_matrix(
            label_img, label_txt, labels=[i for i in range(num)])
        cost_matrix = calculate_cost_matrix(confusion_matrix, num)
        indices = Munkres().compute(cost_matrix)
        kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(
            indices).astype(int)
        return kmeans_to_true_cluster_labels

    def cross_rec(self, z_img, z_txt, index_swap):
        B1 = z_txt.size(0)
        B2 = z_img.size(0)
        z_img_pre, prototype_img, z_txt_pre, prototype_txt = self.pre(
            z_img, self.prototype_img, z_txt, self.prototype_txt)
        z_img, c_img, attn_img, v_img, _ = self.cross_img(
            z_img_pre, prototype_img)
        z_txt, c_txt, attn_txt, v_txt, _ = self.cross_txt(
            z_txt_pre, prototype_txt)

        attn_img = attn_img[:, :, index_swap]
        rev_swap = np.argsort(index_swap)
        attn_txt = attn_txt[:, :, rev_swap]
        rec_img = rearrange(
            attn_txt @ v_img,
            "h n c -> n (h c)",
            h=self.num_heads,
            n=B1,
            c=self.dim // self.num_heads,
        )

        rec_img = F.normalize(rec_img, dim=1) + F.normalize(z_txt_pre, dim=1)
        rec_txt = rearrange(
            attn_img @ v_txt,
            "h n c -> n (h c)",
            h=self.num_heads,
            n=B2,
            c=self.dim // self.num_heads,
        )
        rec_txt = F.normalize(rec_txt, dim=1) + F.normalize(z_img_pre, dim=1)
        rec_img = self.proj_z(rec_img)
        rec_txt = self.proj_z(rec_txt)
        rec_img = self.l2_norm(rec_img, dim=1)
        rec_txt = self.l2_norm(rec_txt, dim=1)
        return rec_img, rec_txt

    def eval_z(self, z_img, z_txt, index_swap, epoch):
        z_img_pre, prototype_img, z_txt_pre, prototype_txt = self.pre(
            z_img, self.prototype_img, z_txt, self.prototype_txt)
        z_img, c_img, attn_img, _, _ = self.cross_img(z_img_pre, prototype_img)
        z_txt, c_txt, attn_txt, _, _ = self.cross_txt(z_txt_pre, prototype_txt)
        z_img = F.normalize(z_img, dim=1) + F.normalize(z_img_pre, dim=1)
        z_txt = F.normalize(z_txt, dim=1) + F.normalize(z_txt_pre, dim=1)
        z_img = self.proj_z(z_img)
        z_txt = self.proj_z(z_txt)
        z_img = self.l2_norm(z_img, dim=1)
        z_txt = self.l2_norm(z_txt, dim=1)

        return z_img, z_txt, c_img, c_txt, attn_img, attn_txt

    def train_z(self, z_img, z_txt):
        z_img_pre, prototype_img, z_txt_pre, prototype_txt = self.pre(
            z_img, self.prototype_img, z_txt, self.prototype_txt)
        z_img, c_img, attn_img, _, _ = self.cross_img(z_img_pre, prototype_img)
        z_txt, c_txt, attn_txt, _, _ = self.cross_txt(z_txt_pre, prototype_txt)
        z_img = F.normalize(z_img, dim=1) + F.normalize(z_img_pre, dim=1)
        z_txt = F.normalize(z_txt, dim=1) + F.normalize(z_txt_pre, dim=1)
        z_img = self.proj_z(z_img)
        z_txt = self.proj_z(z_txt)
        c_img = F.normalize(c_img, dim=1) + F.normalize(prototype_img, dim=1)
        c_txt = F.normalize(c_txt, dim=1) + F.normalize(prototype_txt, dim=1)
        c_img = self.proj_c(c_img)
        c_txt = self.proj_c(c_txt)

        return z_img, z_txt, c_img, c_txt, attn_img, attn_txt

    def dual_pre(self, z_img, z_txt, index_swap):
        rec_img, rec_txt = self.cross_rec(z_img, z_txt, index_swap)
        return rec_txt, rec_img

    def forward(self, z_img, z_txt, index_swap):
        z_img, z_txt, c_img, c_txt, attn_img, attn_txt = self.train_z(
            z_img, z_txt)
        loss_ins = self.ins(self.l2_norm(self.projector1(
            z_img), dim=1), self.l2_norm(self.projector1(z_txt), dim=1))

        if index_swap is None:
            loss_prototype = torch.tensor(0)
        else:
            c_img = c_img[index_swap, :]
            prototype_cat_1 = torch.cat([c_txt, c_img], dim=0)
            prototype_cat_2 = torch.cat([c_img, c_txt], dim=0)
            loss_prototype = self.pro(self.l2_norm(self.projector2(prototype_cat_1), dim=1),
                                      self.l2_norm(self.projector2(prototype_cat_2), dim=1))

        attn_img = attn_img.mean(0)
        attn_txt = attn_txt.mean(0)
        loss_reg = attention_reg(attn_img, attn_txt)
        loss_ins += loss_reg
        return loss_ins, loss_prototype


def attention_reg(c_i, c_j):
    p_i = c_i.sum(0).view(-1)
    p_i /= p_i.sum()
    p_j = c_j.sum(0).view(-1)
    p_j /= p_j.sum()
    ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
    ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
    loss_ne = (ne_i + ne_j)/2

    ne_i_onehot = -torch.sum(c_i * torch.log(c_i + 1e-8), dim=1).mean()
    ne_j_onehot = -torch.sum(c_j * torch.log(c_j + 1e-8), dim=1).mean()
    loss_onehot = (ne_i_onehot + ne_j_onehot)*0.02/2
    return loss_ne + loss_onehot


class cross_attn(nn.Module):
    def __init__(self, dim=128, num=10, head=1):
        super(cross_attn, self).__init__()
        self.scale = dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj1 = nn.Linear(dim, dim, bias=True)
        self.v_proj2 = nn.Linear(dim, dim, bias=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.num_heads = head
        self.dim = dim
        self.num = num
        self.proj_z = nn.Identity()
        self.proj_c = nn.Identity()

    def forward(self, z, prototype):
        B = z.size(0)
        q = rearrange(
            self.q_proj(self.norm1(z)),
            "n (h c)-> h n c",
            h=self.num_heads,
            n=B,
            c=self.dim // self.num_heads,
        )
        k = rearrange(
            self.k_proj(self.norm2(prototype)),
            "n (h c)-> h n c",
            n=self.num,
            h=self.num_heads,
            c=self.dim // self.num_heads,
        )
        v = rearrange(
            self.v_proj1(self.norm2(prototype)),
            "n (h c)-> h n c",
            n=self.num,
            h=self.num_heads,
            c=self.dim // self.num_heads,
        )
        v1 = rearrange(
            self.v_proj2(self.norm1(z)),
            "n (h c)-> h n c",
            h=self.num_heads,
            n=B,
            c=self.dim // self.num_heads,
        )
        sim = (q @ k.transpose(-2, -1)) * self.scale
        attn = sim.softmax(dim=-1)
        z_c = rearrange(
            attn @ v,
            "h n c -> n (h c)",
            h=self.num_heads,
            n=B,
            c=self.dim // self.num_heads,
        )
        c_z = rearrange(
            attn.transpose(-2, -1) @ v1,
            "h n c -> n (h c)",
            h=self.num_heads,
            n=self.num,
            c=self.dim // self.num_heads,
        )
        z_c = self.proj_z(z_c)
        c_z = self.proj_c(c_z)
        return z_c, c_z, attn, v, z


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels
