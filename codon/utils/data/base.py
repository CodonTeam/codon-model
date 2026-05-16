import random
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

import codon
from torch.utils.data import DataLoader, Dataset


@runtime_checkable
class Stateful(Protocol):
    '''
    Structural protocol for objects whose internal state can be persisted to
    a checkpoint and later restored.

    Any object that implements both `state_dict()` and `load_state_dict()`
    matches this protocol.
    '''

    def state_dict(self) -> Dict[str, Any]:
        '''
        Return a dictionary representing the object's serializable state.

        Returns:
            Dict[str, Any]: A picklable mapping of the object's state.
        '''
        ...

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        '''
        Restore the object's state from a previously produced state dict.

        Args:
            state (Dict[str, Any]): The state dictionary produced by
                `state_dict()`.
        '''
        ...


class TorchDatasetWrapper(Dataset):
    '''
    A wrapper class to convert a CodonDataset into a PyTorch compatible Dataset.

    Attributes:
        dataset (CodonDataset): The underlying CodonDataset instance.
        collate_fn (Optional[Callable]): An optional function to process or format
            the dataset items (e.g., converting dataclasses to tuples).
        _indices (Optional[List[int]]): An optional list of shuffled indices.
        _seek_offset (int): The current sequential read offset.
    '''

    def __init__(self, dataset: 'CodonDataset', collate_fn: Optional[Callable] = None) -> None:
        '''
        Initializes the TorchDatasetWrapper.

        Args:
            dataset (CodonDataset): The dataset to wrap.
            collate_fn (Optional[Callable]): Optional function applied to each item.
        '''
        self.dataset = dataset
        self.collate_fn = collate_fn
        self._indices: Optional[List[int]] = None
        self._seek_offset: int = 0

    def shuffle(self, seed: Optional[int] = None) -> 'TorchDatasetWrapper':
        '''
        Shuffles the underlying dataset indices.

        Args:
            seed (Optional[int]): The random seed. If None, uses codon.__seed__ or 42.

        Returns:
            TorchDatasetWrapper: self for method chaining.
        '''
        if seed is None:
            seed = codon.__seed__ if codon.__seed__ is not None else 42
            
        rng = random.Random(seed)
        self._indices = list(range(len(self.dataset)))
        rng.shuffle(self._indices)
        return self

    def seek(self, offset: int) -> 'TorchDatasetWrapper':
        '''
        Sets the sequential read offset for the dataset.

        Args:
            offset (int): The number of items to skip.

        Returns:
            TorchDatasetWrapper: self for method chaining.
        '''
        self._seek_offset = max(0, offset)
        return self

    def state_dict(self) -> Dict[str, Any]:
        '''
        Returns a dictionary containing the state of the dataset wrapper.

        Returns:
            Dict[str, Any]: The state dictionary.
        '''
        return {
            'indices': self._indices,
            'seek_offset': self._seek_offset
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        '''
        Restores the state of the dataset wrapper.

        Args:
            state (Dict[str, Any]): The state dictionary to restore from.
        '''
        self._indices = state.get('indices', None)
        self._seek_offset = state.get('seek_offset', 0)

    def __len__(self) -> int:
        '''
        Returns the length of the dataset.

        Returns:
            int: The total number of items minus the seek offset.
        '''
        return max(0, len(self.dataset) - self._seek_offset)

    def __getitem__(self, idx: int) -> Any:
        '''
        Retrieves an item from the dataset at the specified index,
        applying the collate_fn if provided and considering seek offset and shuffle.

        Args:
            idx (int): The index of the item.

        Returns:
            Any: The potentially processed item.
        '''
        actual_idx = idx + self._seek_offset
        if self._indices is not None:
            actual_idx = self._indices[actual_idx]
            
        item = self.dataset[actual_idx]
        if self.collate_fn is not None:
            return self.collate_fn(item)
        return item

    def loader(self, batch_size: int = 1, shuffle: bool = False, **kwargs: Any) -> DataLoader:
        '''
        Creates a PyTorch DataLoader from this wrapped dataset.

        Args:
            batch_size (int): How many samples per batch to load. Defaults to 1.
            shuffle (bool): Set to True to have the data reshuffled at every epoch. Defaults to False.
            **kwargs: Additional keyword arguments to pass to DataLoader (e.g., num_workers, pin_memory).

        Returns:
            DataLoader: A PyTorch DataLoader instance.
        '''
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)


class CodonDataset:
    '''
    Base class for all Codon datasets.

    This abstract class defines the interface that all datasets must implement.
    It provides a common structure for accessing data rows and length.
    '''

    @property
    def row(self) -> int:
        '''
        Returns the number of rows in the dataset.

        Returns:
            int: The total number of rows.
        '''
        return len(self)

    def __len__(self) -> int:
        '''
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        '''
        raise NotImplementedError

    def __getitem__(self, idx: Any) -> Any:
        '''
        Retrieves an item from the dataset at the specified index.

        Args:
            idx (Any): The index of the item to retrieve.

        Returns:
            Any: The item at the specified index.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        '''
        raise NotImplementedError

    def compose(self, collate_fn: Optional[Callable] = None, shuffle: bool = False, seed: Optional[int] = None, seek: int = 0, **kwargs: Any) -> TorchDatasetWrapper:
        '''
        Wraps the dataset into a PyTorch compatible Dataset.

        Args:
            collate_fn (Optional[Callable]): An optional function to process each item
                returned by __getitem__ (e.g., to convert dataclasses to tuples).
            shuffle (bool): Whether to shuffle the dataset. Defaults to False.
            seed (Optional[int]): Random seed for shuffling. Defaults to None.
            seek (int): The initial read offset. Defaults to 0.
            **kwargs: Additional kwargs.

        Returns:
            TorchDatasetWrapper: A PyTorch Dataset instance wrapping this CodonDataset.
        '''
        wrapper = TorchDatasetWrapper(self, collate_fn)
        if shuffle:
            wrapper.shuffle(seed)
        if seek > 0:
            wrapper.seek(seek)
        return wrapper
