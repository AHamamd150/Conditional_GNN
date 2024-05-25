# Generic imports
import scipy
import numpy as np
import pandas as pd
import sklearn.preprocessing as sklp

# PyTorch imports
import torch
from torch.utils.data import random_split

# PyTorch Geometric imports
import torch_geometric
import torch_geometric.data as PyGData
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

# PyTorch Lightning imports
from lightning.pytorch import LightningDataModule


class HiggsHeteroDataModule(LightningDataModule):
    def __init__(
        self,
        name: str,
        seed: int,
        shuffle: bool = True,
        num_features: int = 12,
        num_nodes: int = 12,
        batch_size: int = 32,
        num_workers: int = 16,
    ) -> None:
        super().__init__()

        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_features = num_features
        self.num_nodes = num_nodes

        # save the parameters
        self.save_hyperparameters()

        self.graphs = pd.read_pickle(name)
        if shuffle:
            self.graphs = self.graphs.sample(frac=1.0)

        self.scale_particle = np.array([1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 1.0, 1.0])

    def angular_distance(self, eta1, phi1, eta2, phi2):
        d_eta = eta1 - eta2
        d_phi = phi1 - phi2
        d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
        return np.sqrt(d_eta**2 + d_phi**2)

    def __getitem__(self, item) -> tuple:
        graph = self.graphs.iloc[item].to_numpy()
        x, y = graph.reshape(self.num_nodes, self.num_features), graph[-11]
        # Initialize a HeteroData object
        data = torch_geometric.data.HeteroData()

        # Type-I nodes: particles: node features
        particle_x = x[:-1, :8] / self.scale_particle

        # Type-II nodes: event node: node features
        event_x = x[-1:, [2, 3, 8, 9, 10, 11]]  # TODO: scale this

        # Type-III nodes: For Parameterized ML
        data["parameter"].x = torch.tensor(x[-1:, 0:1], dtype=torch.float32)

        # Convert to torch tensors
        data["particle"].x = torch.tensor(particle_x, dtype=torch.float32)
        data["event"].x = torch.tensor(event_x, dtype=torch.float32)

        # Calculate angular separations and create fully connected edges
        num_particles = particle_x.shape[0]
        edge_index = []
        edge_weight = []
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                delta_r = self.angular_distance(
                    particle_x[i, 6],
                    particle_x[i, 7],
                    particle_x[j, 6],
                    particle_x[j, 7],
                )
                edge_index.append([i, j])
                edge_index.append([j, i])  # Add reverse edge for undirected graph
                edge_weight.extend([delta_r, delta_r])  # Duplicate for reverse edge
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

        # Nodes in order
        # 0 = lepton1
        # 1 = lepton2
        # 2 = bjet1
        # 3 = bjet2
        # 4 = bjet3
        # 5 = bjet4
        # 6 = neutrino1
        # 7 = neutrino2
        # 8 = top1
        # 9 = top2
        # 10 = higgs
        # 11 = event

        data["particle", "connects", "particle"].edge_index = edge_index
        data["particle", "connects", "particle"].edge_attr = edge_weight
        """
        # Graph convectivity:
        # Connect bjet3 and bjet4 to the higgs
        higgs_edges = torch.tensor([[4, 5], [10, 10]], dtype=torch.long)
        # Connect bjet1, lepton1, and neutrino1 to top1
        top1_edges = torch.tensor([[0, 2, 6], [8, 8, 8]], dtype=torch.long)
        # Connect bjet2, lepton2, and neutrino2 to top2
        top2_edges = torch.tensor([[1, 3, 7], [9, 9, 9]], dtype=torch.long)
        # Assign to HeteroData
        data["particle", "connects", "particle"].edge_index = higgs_edges
        data["particle", "connects", "particle"].edge_index = torch.cat(
            [top1_edges, top2_edges], dim=1
        )
        """
        data["particle", "connects", "event"].edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )

        data["event", "connects", "parameter"].edge_index = torch.tensor(
            [
                [0],
                [0],
            ],
            dtype=torch.long,
        )

        data = T.ToUndirected()(data)
        y = torch.tensor(y, dtype=torch.long)

        return (data, y)

    def __len__(self) -> int:
        return len(self.graphs)

    def setup(self, stage: str) -> None:
        dataset = self

        # TODO: remove hardcoded numbers: fractions
        self.train, self.val, self.test = random_split(
            dataset, [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=False,
        )


class HiggsHomoDataModule(LightningDataModule):
    def __init__(
        self,
        name: str,
        seed: int,
        shuffle: bool = True,
        num_features: int = 12,
        num_nodes: int = 12,
        batch_size: int = 32,
        num_workers: int = 16,
    ) -> None:
        super().__init__()

        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_features = num_features
        self.num_nodes = num_nodes

        # save the parameters
        self.save_hyperparameters()

        self.graphs = pd.read_pickle(name)
        if shuffle:
            self.graphs = self.graphs.sample(frac=1.0)

        # self.labels = self.graphs["label"].to_numpy()
        # classes, counts = np.unique(self.labels, return_counts=True)
        # weights = 1.0 / counts.float()
        # sample_weights = weights[y_train_tensor.squeeze().long()]

        self.scale_particle = np.array([1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 1.0, 1.0])

    def angular_distance(self, eta1, phi1, eta2, phi2):
        d_eta = eta1 - eta2
        d_phi = phi1 - phi2
        d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
        return np.sqrt(d_eta**2 + d_phi**2)

    def __getitem__(self, item) -> tuple:
        graph = self.graphs.iloc[item].to_numpy()
        x, y = graph.reshape(self.num_nodes, self.num_features), graph[-11]

        # Type-I nodes: particles: node features
        particle_x = torch.tensor(x[:-1, :8] / self.scale_particle, dtype=torch.float32)

        # High Level variables (angular distributions)
        x1 = (x[:-1, 4:8] / np.array([100.0, 100.0, 1.0, 1.0])).flatten()
        x2 = x[-1, [2, 3, 8, 9, 10, 11]]
        event_x = torch.tensor(np.concatenate([x1, x2]), dtype=torch.float32)

        # The parameter
        parameter_x = torch.tensor(x[-1:, 0], dtype=torch.float32)

        # edge index and edge attributes
        # compute pair-wise distance matrix from (eta, phi)
        hitPosMatrix = x[:-1, [6, 7]]
        hitDistMatrix = scipy.spatial.distance_matrix(hitPosMatrix, hitPosMatrix)
        norm_hitDistMatrix = sklp.normalize(hitDistMatrix)
        norm_sparse_hitDistMatrix = scipy.sparse.csr_matrix(norm_hitDistMatrix)
        hitEdges = torch_geometric.utils.convert.from_scipy_sparse_matrix(
            norm_sparse_hitDistMatrix
        )
        """
        # Calculate angular separations and create fully connected edges
        num_particles = particle_x.shape[0]
        edge_index = []
        edge_weight = []
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                delta_r = self.angular_distance(
                    particle_x[i, 6],
                    particle_x[i, 7],
                    particle_x[j, 6],
                    particle_x[j, 7],
                )
                edge_index.append([i, j])
                edge_index.append([j, i])  # Add reverse edge for undirected graph
                edge_weight.extend([delta_r, delta_r])  # Duplicate for reverse edge
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        """
        # The graph
        G = PyGData.Data(
            x=particle_x,
            y=torch.tensor(y, dtype=torch.long),
            edge_index=hitEdges[0],
            edge_attr=hitEdges[1].float(),
        )

        return (G, event_x, parameter_x)

    def __len__(self) -> int:
        return len(self.graphs)

    def setup(self, stage: str) -> None:
        dataset = self

        # TODO: remove hardcoded numbers: fractions
        self.train, self.val, self.test = random_split(
            dataset, [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=False,
        )
