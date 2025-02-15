#!/usr/bin/env python3

import abc
import torch
from typing import List
from itertools import product


class BaseIndex(abc.ABC):
    """
    Provides a base interface for blocking data and establishing neighbor relationships.
    Cannot be directly instantiated and must be subclassed before use.

    Subclasses must implement the set_blocks, set_neighbors, and set_test_blocks methods. Use help() to learn more
    about what these methods must return.

    :param set_blocks_kwargs: Dict of keyword arguments to be passed to child's set_blocks implementation.
    :param set_neighbors_kwargs: Dict of keyword arguments to be passed to child's set_neighbors implementation.
    :param device: Optional torch device. If None, will use CUDA if available, else CPU.
    """

    def __init__(
            self,
            set_blocks_kwargs: dict = None,
            set_neighbors_kwargs: dict = None,
            device: torch.device = None
    ):
        self._device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._block_observations = None
        self._neighbor_blocks = None
        self._exclusive_neighbor_observations = None
        self._inclusive_neighbor_observations = None
        self._test_block_observations = None

        self.set_blocks_kwargs = set_blocks_kwargs
        self.set_neighbors_kwargs = set_neighbors_kwargs

        if self.set_blocks_kwargs is None:
            self._block_observations = self.set_blocks()
        else:
            self._block_observations = self.set_blocks(**self.set_blocks_kwargs)

        # create 1D tensors out of any 0D tensors and ensure device placement
        self._block_observations = [
            (block.reshape(1) if block.dim() == 0 else block).to(self._device)
            for block in self._block_observations
        ]

        # complete template method by creating ordered neighbors
        self._create_ordered_neighbors()

    def _create_ordered_neighbors(self):
        """
        Calculates neighboring relationships based on the order defined by self._block_observations, using the algorithm
        defined by self.set_neighbors(). Since the results of these calculations implicitly depend on the order of
        self._block_observations, we wrap these steps in a separate function, so we can recalculate if the order of
        self._block_observations changes via a call to self.reorder().
        """
        if self.set_neighbors_kwargs is None:
            self._neighbor_blocks = self.set_neighbors()
        else:
            self._neighbor_blocks = self.set_neighbors(**self.set_neighbors_kwargs)

        # pre-allocating lists is more efficient than using .append
        n_blocks = len(self._block_observations)
        exclusive_neighboring_observations = [None] * n_blocks
        inclusive_neighboring_observations = [None] * n_blocks

        for i in range(n_blocks):
            if len(self._neighbor_blocks[i]) == 0:
                exclusive_neighboring_observations[i] = torch.LongTensor([]).to(self._device)
                inclusive_neighboring_observations[i] = self._block_observations[i]
            else:
                neighbor_size = sum(len(self._block_observations[j])
                                  for j in self._neighbor_blocks[i])
                neighbors_tensor = torch.empty(neighbor_size,
                                            dtype=self._block_observations[0].dtype,
                                            device=self._device)

                pos = 0
                for j in self._neighbor_blocks[i]:
                    block_size = len(self._block_observations[j])
                    neighbors_tensor[pos:pos + block_size] = self._block_observations[j]
                    pos += block_size

                exclusive_neighboring_observations[i] = neighbors_tensor

                inclusive_size = len(self._block_observations[i]) + neighbor_size
                inclusive_tensor = torch.empty(inclusive_size,
                                               dtype=self._block_observations[0].dtype,
                                               device=self._device)
                inclusive_tensor[:len(self._block_observations[i])] = self._block_observations[i]
                inclusive_tensor[len(self._block_observations[i]):] = neighbors_tensor

                inclusive_neighboring_observations[i] = inclusive_tensor

        self._exclusive_neighbor_observations = exclusive_neighboring_observations
        self._inclusive_neighbor_observations = inclusive_neighboring_observations

    def _reorder(self, new_order: torch.LongTensor):
        """
        Reorders self._block_observations to the order specified by new_order. The ordered neighbors are recalculated,
        and all the relevant lists are modified in place.

        :param new_order: Tensor where the ith element contains the index of the block to be moved to index i.
        """
        self._block_observations = [self._block_observations[idx] for idx in new_order]
        # neighboring blocks get recomputed here based on new ordering, so we do not have to explicitly reorder them
        self._create_ordered_neighbors()

    @abc.abstractmethod
    def set_blocks(self, **kwargs) -> List[torch.LongTensor]:
        """
        Creates a list of length equal to the number of blocks, where the ith element is a tensor containing the
        indices of the training set that belong to the ith block.

        :param kwargs: Keyword arguments to be passed to child's set_blocks implementation.
        """
        ...

    @abc.abstractmethod
    def set_neighbors(self, **kwargs) -> List[torch.LongTensor]:
        """
        Creates a list of length equal to the number of blocks, where the ith element is a tensor containing the
        indices of the blocks that neighbor the ith block.

        :param kwargs: Keyword arguments to be passed to child's set_neighbors implementation.
        """
        ...

    @abc.abstractmethod
    def set_test_blocks(self, *args, **kwargs) -> List[torch.LongTensor]:
        """
        Creates a list of length equal to the number of blocks, where the ith element is a tensor containing the
        indices of the testing set that belong to the ith block.

        :param args: Positional arguments to be passed to child's set_test_blocks implementation.
        :param kwargs: Keyword arguments to be passed to child's set_test_blocks implementation.
        """
        ...

    @abc.abstractmethod
    def reorder(self, *args, **kwargs):
        """
        In-place operation that defines a new ordering of blocks (new_order) in terms of their indices, reorders any
        order-dependent quantities that the child class defines, and calls super()._reorder(new_order).
        """
        ...

    @property
    def blocks(self) -> List[torch.LongTensor]:
        """
        List of tensors where the ith tensor contains the indices of the training set points belonging to block i.
        """
        return self._block_observations

    @property
    def neighbors(self) -> List[torch.LongTensor]:
        """
        List of tensors, where the ith tensor contains the indices of the training set points belonging to the neighbor
        set of block i.
        """
        return self._exclusive_neighbor_observations

    @property
    def test_blocks(self) -> List[torch.LongTensor]:
        """
        List of tensors where the ith tensor contains the indices of the testing set points belonging to block i.
        Only defined after block_new_data has been called.
        """
        if self._test_block_observations is None:
            raise RuntimeError(
                "Blocks of testing data do not exist, as the 'block_new_data' "
                "method has not been called on testing data."
            )
        return self._test_block_observations

    @property
    def test_neighbors(self) -> List[torch.LongTensor]:
        """
        List of tensors, where the ith tensor contains the indices of the training set points belonging to the
        neighbor set of the ith test block. Importantly, the neighbor sets of test blocks only consist of training
        points. Only defined after block_new_data has been called.
        """
        if self._test_block_observations is None:
            raise RuntimeError(
                "Neighboring sets of testing blocks do not exist, as the 'block_new_data' "
                "method has not been called on testing data."
            )
        return self._inclusive_neighbor_observations

    def block_new_data(self, *args, **kwargs) -> None:
        """
        Calls the set_test_blocks method defined by the child class.
        """
        test_block_observations = self.set_test_blocks(*args, **kwargs)

        # create 1D tensors out of any 0D tensors
        self._test_block_observations = [
            block.reshape(1) if block.dim() == 0 else block for block in test_block_observations
        ]


class BaseBatchIndex(abc.ABC):

    def __init__(
            self,
            batch_shape: torch.Size,
            create_batch_index_kwargs: dict = None,
            device: torch.device = None
    ):
        self._device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._batch_shape = batch_shape
        # cartesian product of possible batch indices and dict for easy access
        self._batch_idx_list = list(product(*[[_ for _ in range(i)] for i in self._batch_shape]))
        self._batch_dict = dict(zip(self._batch_idx_list, range(len(self._batch_idx_list))))

        if create_batch_index_kwargs is None:
            self._index = self.create_batch_index()
        else:
            self._index = self.create_batch_index(**create_batch_index_kwargs)

    def _verify_batch_indices(self, *batch_indices: int):
        """
        Helper function for validating user input of batch indices, ensuring the correct shape and valid range.
        """
        if len(batch_indices) != len(self._batch_shape):
            raise ValueError(
                f"Number of batch indices provided ({len(batch_indices)}) does not match the number of dimensions in "
                f"the batch shape ({len(self._batch_shape)})"
            )

        for i, batch_idx in enumerate(batch_indices):
            if batch_idx >= self._batch_shape[i]:
                raise ValueError(
                    f"Batch index {batch_idx} is out of range for dimension {i} of the batch shape ({self._batch_shape})"
                )

    @abc.abstractmethod
    def create_batch_index(self, *args, **kwargs) -> List[BaseIndex]:
        ...

    @property
    def batch_shape(self) -> torch.Size:
        """
        Batch shape, where each element is the number of batches in the corresponding batch dimension.
        """
        return self._batch_shape

    @property
    def batch_idx_list(self):
        """
        List of all possible batch indices, where each element is a tuple of possible batch indices. This is primarily
        used for accessing the index object for a specific batch.
        """
        return self._batch_idx_list

    def get_index_for_batch(self, *batch_indices: int) -> BaseIndex:
        """
        Returns the index object for the batch specified by the arguments.

        :param batch_indices: Indices of the batch to access. Must be in the range of the batch_shape, and the number of
        indices must match the number of dimensions in the batch_shape.
        """
        self._verify_batch_indices(*batch_indices)
        return self._index[self._batch_dict[batch_indices]]

    def blocks(self, *batch_indices: int) -> List[torch.LongTensor]:
        """
        List of tensors where the ith tensor contains the indices of the training set points belonging to block i, in
        the batch specified by the arguments.

        :param batch_indices: Indices of the batch to access. Must be in the range of the batch_shape, and the number of
        indices must match the number of dimensions in the batch_shape.
        """
        self._verify_batch_indices(*batch_indices)
        return self.get_index_for_batch(*batch_indices).blocks

    def neighbors(self, *batch_indices: int) -> List[torch.LongTensor]:
        """
        List of tensors, where the ith tensor contains the indices of the training set points belonging to the neighbor
        set of block i, in the batch specified by the arguments.

        :param batch_indices: Indices of the batch to access. Must be in the range of the batch_shape, and the number of
        indices must match the number of dimensions in the batch_shape.
        """
        self._verify_batch_indices(*batch_indices)
        return self.get_index_for_batch(*batch_indices).neighbors

    def test_blocks(self, *batch_indices: int) -> List[torch.LongTensor]:
        """
        List of tensors where the ith tensor contains the indices of the testing set points belonging to block i, in
        the batch specified by the arguments.

        :param batch_indices: Indices of the batch to access. Must be in the range of the batch_shape, and the number of
        indices must match the number of dimensions in the batch_shape.
        """
        self._verify_batch_indices(*batch_indices)
        return self.get_index_for_batch(*batch_indices).test_blocks

    def test_neighbors(self, *batch_indices: int) -> List[torch.LongTensor]:
        """
        List of tensors, where the ith tensor contains the indices of the training set points belonging to the neighbor
        set of block i, in the batch specified by the arguments.

        :param batch_indices: Indices of the batch to access. Must be in the range of the batch_shape, and the number of
        indices must match the number of dimensions in the batch_shape.
        """
        self._verify_batch_indices(*batch_indices)
        return self.get_index_for_batch(*batch_indices).test_neighbors
