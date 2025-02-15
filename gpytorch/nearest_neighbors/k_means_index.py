#!/usr/bin/env python3

from typing import List, Callable
import warnings
import torch

from .index import BaseIndex, BaseBatchIndex


class KMeansIndex(BaseIndex):
    """
    This index performs K-Means clustering on a given feature set, computes neighboring blocks, enables
    evaluating block membership for test points, and enables reordering of the blocks based on block centroids.

    :param data: Features to cluster via K-Means, typically an n x 2 tensor of spatial lat-long coordinates.
    :param n_blocks: Number of desired clusters. Note that this does not guarantee similarly-sized clusters.
    :param n_neighbors: Number of neighboring clusters per cluster.
    """

    def __init__(
            self,
            data: torch.tensor,
            n_blocks: int,
            n_neighbors: int,
            distance_metric: Callable,
            preferred_nnlib="faiss",
            device: torch.device = None
    ):
        self.n_blocks = n_blocks
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.centroids = None

        # Handle nnlib selection
        if preferred_nnlib == "faiss":
            try:
                import faiss
                self.nnlib = "faiss"
            except ImportError:
                warnings.warn(
                    "Tried to import faiss, but failed. Falling back to scikit-learn nearest neighbor search.",
                    ImportWarning,
                )
                self.nnlib = "sklearn"
        else:
            self.nnlib = "sklearn"

        # Move data to device before passing to parent
        data = data.to(device) if device is not None else data
        super().__init__(set_blocks_kwargs={"data": data}, set_neighbors_kwargs={}, device=device)

    def _get_cluster_membership(self, data: torch.tensor) -> List[torch.LongTensor]:
        """
        Determines which K-Means cluster each point in the provided data belongs to.

        :param data: Tensor for which to evaluate cluster membership. If any of these points are outside the domain
            of the points used to train the K-Means clusters, you may get nonsensical results.

        :return: List of tensors, where the ith tensor contains the indices of the points in data that belong to the
            ith K-Means cluster.
        """
        n_blocks = len(self.centroids)
        blocks = [None] * n_blocks

        # Compute all distances at once
        distances = torch.cdist(data, self.centroids)
        block_per_point = distances.argmin(dim=1)

        # Use boolean indexing instead of nonzero()
        for block in range(n_blocks):
            mask = block_per_point == block
            blocks[block] = torch.arange(len(data))[mask]

        return blocks

    def set_blocks(self, data: torch.tensor) -> List[torch.LongTensor]:

        if self.nnlib == "faiss":
            from faiss import Kmeans
            kmeans = Kmeans(data.shape[1], self.n_blocks, niter=10)
            kmeans.train(data)

            # k-means gives centroids directly, so save centroids
            self.centroids = torch.tensor(kmeans.centroids)

        else:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_blocks, n_init=10, max_iter=300)
            kmeans.fit(data.numpy())  # Convert tensor to numpy array

            # Extract centroids and convert back to tensor
            self.centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        return self._get_cluster_membership(data)

    def set_neighbors(self) -> List[torch.LongTensor]:
        # if there are no neighbors, we want a list of empty tensors
        if self.n_neighbors == 0:
            return [torch.LongTensor([]) for _ in range(0, self.n_blocks)]

        else:
            # get distance matrix and find ordered distances
            sorter = self.distance_metric(self.centroids, self.centroids).argsort().long()
            return [sorter[i][sorter[i] < i][0 : self.n_neighbors] for i in range(0, len(sorter))]

    def set_test_blocks(self, new_data: torch.tensor) -> List[torch.LongTensor]:
        return self._get_cluster_membership(new_data)

    def reorder(self, ordering_strategy):
        # new order is defined as some reordering of the K-means block centroids
        new_order = ordering_strategy(self.centroids)

        # reorder the instance attributes that depend on the ordering
        self.centroids = self.centroids[new_order]

        # reorder superclass attributes and recompute neighbors under new ordering
        super()._reorder(new_order)


class BatchKMeansIndex(BaseBatchIndex):

    def __init__(
            self,
            data: torch.tensor,
            batch_shape: torch.Size,
            n_blocks: int,
            n_neighbors: int,
            distance_metric: Callable,
            preferred_nnlib="faiss"
    ):
        super(BatchKMeansIndex, self).__init__(batch_shape=batch_shape, create_batch_index_kwargs={
            "data": data,
            "n_blocks": n_blocks,
            "n_neighbors": n_neighbors,
            "distance_metric": distance_metric,
            "preferred_nnlib": preferred_nnlib
        })

    def create_batch_index(
            self,
            data: torch.tensor,
            n_blocks: int,
            n_neighbors: int,
            distance_metric: Callable,
            preferred_nnlib: str
    ) -> List[KMeansIndex]:

        indexes = []
        for idx in self.batch_idx_list:
            indexes.append(KMeansIndex(data[idx], n_blocks, n_neighbors, distance_metric, preferred_nnlib))

        return indexes
