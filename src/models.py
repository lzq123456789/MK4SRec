# -*- coding:utf-8 -*-

import os
import math
import copy
import pickle
import gensim
import random
import torch
import torch.nn as nn
from ss import KAN
from tqdm import tqdm
from modules import TransformerEncoder, get_attention_mask
from mamba_ssm import Mamba
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import torch.nn as nn
import torch.optim as optim
import numpy as np

# n_center: The number of centers, e.g., 15.
# feature_maps_shape: The shape of input feature maps (channel, width, height), e.g., [32, 16, 16].
# num_classes: The number of classes, e.g., 10.
# contrastive_module_dim: The dimention of the contrastive module, e.g., 256.
# head_class_lists: The index of head classes, e.g., [0, 1, 2].
# transfer_strength: Transfer strength, e.g., 1.0.
# epoch_thresh: The epoch index when rare-class samples are generated: e.g., 159.

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(MLP, self).__init__()

        self.num_layers = num_layers
        # 定义第一层
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        # 定义中间层
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        # 定义最后一层
        self.layers.append(nn.Linear(hidden_size, output_size))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.dropout(self.activation(self.layers[i](x)))

        # 最后一层不需要激活函数和dropout
        x = self.layers[-1](x)

        return x
class KMeans(nn.Module):
    def __init__(self, num_cluster, seed, hidden_size, input_size, output_size,gpu_id=0, device="cpu"):
        super(KMeans, self).__init__()
        self.num_cluster = num_cluster
        self.hidden_size = hidden_size
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        self.gpu_id = 0
        self.prototype_embeddings = nn.Parameter(torch.randn(num_cluster, hidden_size, device=self.device))
        self.kan = KAN(
            layers_hidden=[128, 128, 128],
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            device=self.device
        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)



    def initialize_centroids(self, X):
        """Simplified K-means++ initialization"""
        indices = np.random.choice(X.shape[0], self.num_cluster, replace=False)
        centroids = X[indices]
        self.prototype_embeddings.data = torch.tensor(centroids, device=self.device, dtype=torch.float32)
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        seq_embeddings = self.kan(x)
        similarity = torch.cosine_similarity(seq_embeddings.unsqueeze(1), self.prototype_embeddings.unsqueeze(0),
                                             dim=-1)
        cluster_assignments = similarity.argmax(dim=1)
        return seq_embeddings, cluster_assignments

    def loss_function(self, seq_embeddings, cluster_assignments):
        # Clustering loss: minimize intra-cluster distances
        clustering_loss = torch.mean(
            torch.sum((seq_embeddings - self.prototype_embeddings[cluster_assignments]) ** 2, dim=1))

        # Regularization: encourage even distribution of cluster assignments
        assignment_counts = torch.bincount(cluster_assignments, minlength=self.num_cluster)
        distribution_loss = torch.var(assignment_counts.float())

        # Combine losses
        total_loss = clustering_loss + 0.1 * distribution_loss
        return total_loss

    # def loss_function(self, seq_embeddings, cluster_assignments):
    #     clustering_loss = torch.mean(torch.sum((seq_embeddings - self.prototype_embeddings[cluster_assignments])**2, dim=1))
    #     return clustering_loss

    def train(self, X, max_iterations=20, batch_size=1024, tolerance=1e-4):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.initialize_centroids(X.cpu().numpy())

        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=  True)

        prev_loss = float('inf')
        for iteration in range(max_iterations):
            total_loss = 0
            for batch in dataloader:
                self.optimizer.zero_grad()
                seq_embeddings, cluster_assignments = self.forward(batch[0])
                loss = self.loss_function(seq_embeddings, cluster_assignments)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if abs(prev_loss - avg_loss) < tolerance:
                print(f"Converged at iteration {iteration}")
                break
            prev_loss = avg_loss

            if iteration % 5 == 0:
                print(f"Iteration {iteration}, Loss: {avg_loss}")

    def query(self, x):
        with torch.no_grad():
            seq_embeddings, cluster_assignments = self.forward(x)
        return cluster_assignments, self.prototype_embeddings[cluster_assignments]





class Mamba4Rec(nn.Module):
    def __init__(self, args):
        super(Mamba4Rec, self).__init__()

        self.item_size = args.item_size
        self.max_seq_length = args.max_seq_length
        self.hidden_size = args.hidden_size
        self.num_layers = args.n_layers
        self.dropout_prob = args.hidden_dropout_prob
        self.args = args
        # Mamba specific hyperparameters
        self.d_state = getattr(args, 'd_state', 16)
        self.d_conv = getattr(args, 'd_conv', 4)
        self.expand = getattr(args, 'expand', 2)

        self.item_embedding = nn.Embedding(self.item_size, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids):
        sequence_emb = self.item_embedding(input_ids)
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_emb = self.position_embedding(position_ids)
        sequence_emb += position_emb
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        for layer in self.mamba_layers:
            sequence_emb = layer(sequence_emb)

        return sequence_emb

    def calculate_loss(self, input_ids, target_pos, target_neg):
        sequence_output = self.forward(input_ids)
        last_hidden = sequence_output[:, -1, :]

        pos_emb = self.item_embedding(target_pos)
        neg_emb = self.item_embedding(target_neg)

        pos_logits = (last_hidden * pos_emb).sum(dim=-1)
        neg_logits = (last_hidden * neg_emb).sum(dim=-1)

        loss = -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-24).mean()

        return loss

    def predict(self, input_ids, item_ids):
        sequence_output = self.forward(input_ids)
        last_hidden = sequence_output[:, -1, :]
        item_embs = self.item_embedding(item_ids)
        scores = (last_hidden.unsqueeze(1) * item_embs).sum(dim=-1)
        return scores


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class OnlineItemSimilarity:

    def __init__(self, item_size):
        self.item_size = item_size
        self.item_embedding = None
        self.cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.total_item_list = torch.tensor([i for i in range(self.item_size)],
                                            dtype=torch.long).to(self.device)

    def update_embedding_matrix(self, item_embedding):
        self.item_embedding = copy.deepcopy(item_embedding)
        self.base_embedding_matrix = self.item_embedding(self.total_item_list)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item_idx in range(1, self.item_size):
            try:
                item_vector = self.item_embedding(torch.tensor(item_idx).to(self.device)).view(-1, 1)
                item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
                max_score = max(torch.max(item_similarity), max_score)
                min_score = min(torch.min(item_similarity), min_score)
            except:
                continue
        return max_score, min_score

    def most_similar(self, item_idx, top_k=1, with_score=False):
        item_idx = torch.tensor(item_idx, dtype=torch.long).to(self.device)
        item_vector = self.item_embedding(item_idx).view(-1, 1)
        item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
        item_similarity = (self.max_score - item_similarity) / (self.max_score - self.min_score)
        # remove item idx itself
        values, indices = item_similarity.topk(top_k + 1)
        if with_score:
            item_list = indices.tolist()
            score_list = values.tolist()
            if item_idx in item_list:
                idd = item_list.index(item_idx)
                item_list.remove(item_idx)
                score_list.pop(idd)
            return list(zip(item_list, score_list))
        item_list = indices.tolist()
        if item_idx in item_list:
            item_list.remove(item_idx)
        return item_list


class OfflineItemSimilarity:
    def __init__(self, data_file=None, similarity_path=None, model_name='ItemCF', dataset_name='Sports_and_Outdoors'):
        self.dataset_name = dataset_name
        self.similarity_path = similarity_path
        # train_data_list used for item2vec, train_data_dict used for itemCF and itemCF-IUF
        self.train_data_list, self.train_item_list, self.train_data_dict = self._load_train_data(data_file)
        self.model_name = model_name
        self.similarity_model = self.load_similarity_model(self.similarity_path)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item in self.similarity_model.keys():
            for neig in self.similarity_model[item]:
                sim_score = self.similarity_model[item][neig]
                max_score = max(max_score, sim_score)
                min_score = min(min_score, sim_score)
        return max_score, min_score

    def _convert_data_to_dict(self, data):
        """
        split the data set
        testdata is a test data set
        traindata is a train set
        """
        train_data_dict = {}
        for user, item, record in data:
            train_data_dict.setdefault(user, {})
            train_data_dict[user][item] = record
        return train_data_dict

    def _save_dict(self, dict_data, save_path='./similarity.pkl'):
        print("saving data to ", save_path)
        with open(save_path, 'wb') as write_file:
            pickle.dump(dict_data, write_file)

    def _load_train_data(self, data_file=None):
        """
        read the data from the data file which is a data set
        """
        train_data = []
        train_data_list = []
        train_data_set_list = []
        for line in open(data_file).readlines():
            userid, items = line.strip().split(' ', 1)
            # only use training data
            items = items.split(' ')[:-3]
            train_data_list.append(items)
            train_data_set_list += items
            for itemid in items:
                train_data.append((userid, itemid, int(1)))
        return train_data_list, set(train_data_set_list), self._convert_data_to_dict(train_data)

    def _generate_item_similarity(self, train=None, save_path='./'):
        """
        calculate co-rated users between items
        """
        print("getting item similarity...")
        train = train or self.train_data_dict
        C = dict()
        N = dict()

        if self.model_name in ['ItemCF', 'ItemCF_IUF']:
            print("Step 1: Compute Statistics")
            data_iter = tqdm(enumerate(train.items()), total=len(train.items()))
            for idx, (u, items) in data_iter:
                if self.model_name == 'ItemCF':
                    for i in items.keys():
                        N.setdefault(i, 0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1
                elif self.model_name == 'ItemCF_IUF':
                    for i in items.keys():
                        N.setdefault(i, 0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1 / math.log(1 + len(items) * 1.0)
            self.itemSimBest = dict()
            print("Step 2: Compute co-rate matrix")
            c_iter = tqdm(enumerate(C.items()), total=len(C.items()))
            for idx, (cur_item, related_items) in c_iter:
                self.itemSimBest.setdefault(cur_item, {})
                for related_item, score in related_items.items():
                    self.itemSimBest[cur_item].setdefault(related_item, 0)
                    self.itemSimBest[cur_item][related_item] = score / math.sqrt(N[cur_item] * N[related_item])
            self._save_dict(self.itemSimBest, save_path=save_path)
        elif self.model_name == 'Item2Vec':
            # details here: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py
            print("Step 1: train item2vec model")
            item2vec_model = gensim.models.Word2Vec(sentences=self.train_data_list,
                                                    vector_size=20, window=5, min_count=0,
                                                    epochs=100)
            self.itemSimBest = dict()
            total_item_nums = len(item2vec_model.wv.index_to_key)
            print("Step 2: convert to item similarity dict")
            total_items = tqdm(item2vec_model.wv.index_to_key, total=total_item_nums)
            for cur_item in total_items:
                related_items = item2vec_model.wv.most_similar(positive=[cur_item], topn=20)
                self.itemSimBest.setdefault(cur_item, {})
                for (related_item, score) in related_items:
                    self.itemSimBest[cur_item].setdefault(related_item, 0)
                    self.itemSimBest[cur_item][related_item] = score
            print("Item2Vec model saved to: ", save_path)
            self._save_dict(self.itemSimBest, save_path=save_path)

    def load_similarity_model(self, similarity_model_path):
        if not similarity_model_path:
            raise ValueError('invalid path')
        elif not os.path.exists(similarity_model_path):
            print("the similirity dict not exist, generating...")
            self._generate_item_similarity(save_path=self.similarity_path)
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN']:
            with open(similarity_model_path, 'rb') as read_file:
                similarity_dict = pickle.load(read_file)
            return similarity_dict
        elif self.model_name == 'Random':
            similarity_dict = self.train_item_list
            return similarity_dict

    def most_similar(self, item, top_k=1, with_score=False):
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN']:
            """TODO: handle case that item not in keys"""
            if str(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[str(item)].items(), key=lambda x: x[1],
                                                reverse=True)[0:top_k]
                if with_score:
                    return list(
                        map(lambda x: (int(x[0]), (self.max_score - float(x[1])) / (self.max_score - self.min_score)),
                            top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            elif int(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[int(item)].items(), key=lambda x: x[1],
                                                reverse=True)[0:top_k]
                if with_score:
                    return list(
                        map(lambda x: (int(x[0]), (self.max_score - float(x[1])) / (self.max_score - self.min_score)),
                            top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            else:
                item_list = list(self.similarity_model.keys())
                random_items = random.sample(item_list, k=top_k)
                if with_score:
                    return list(map(lambda x: (int(x), 0.0), random_items))
                return list(map(lambda x: int(x), random_items))
        elif self.model_name == 'Random':
            random_items = random.sample(self.similarity_model, k=top_k)
            if with_score:
                return list(map(lambda x: (int(x), 0.0), random_items))
            return list(map(lambda x: int(x), random_items))



class CombinedModel(nn.Module):
    def __init__(self, mamba_args, kmeans_args):
        super().__init__()
        self.mamba_model = Mamba4Rec(mamba_args)
        self.kmeans_model = KMeans(**kmeans_args)


        self.projection = nn.Linear(
            mamba_args.hidden_size,
            kmeans_args.hidden_size
        )

    def forward(self, sequence_input):

        mamba_output = self.mamba_model(sequence_input)

        sequence_representation = mamba_output[:, -1, :]  # 取最后时间步

        kan_input = self.projection(sequence_representation)

        kan_embeddings, cluster_assignments = self.kmeans_model(kan_input)

        return kan_embeddings, cluster_assignments
