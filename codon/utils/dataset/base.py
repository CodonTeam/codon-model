from typing import Any, Callable, Optional

from torch.utils.data import DataLoader, Dataset


class TorchDatasetWrapper(Dataset):
    '''
    A wrapper class to convert a CodonDataset into a PyTorch compatible Dataset.

    Attributes:
        dataset (CodonDataset): The underlying CodonDataset instance.
        collate_fn (Optional[Callable]): An optional function to process or format
            the dataset items (e.g., converting dataclasses to tuples).
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

    def __len__(self) -> int:
        '''
        Returns the length of the dataset.

        Returns:
            int: The total number of items.
        '''
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        '''
        Retrieves an item from the dataset at the specified index,
        applying the collate_fn if provided.

        Args:
            idx (int): The index of the item.

        Returns:
            Any: The potentially processed item.
        '''
        item = self.dataset[idx]
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

    def compose(self, collate_fn: Optional[Callable] = None, **kwargs: Any) -> TorchDatasetWrapper:
        '''
        Wraps the dataset into a PyTorch compatible Dataset.

        Args:
            collate_fn (Optional[Callable]): An optional function to process each item
                returned by __getitem__ (e.g., to convert dataclasses to tuples).
            **kwargs: Additional kwargs.

        Returns:
            Dataset: A PyTorch Dataset instance wrapping this CodonDataset.
        '''
        return TorchDatasetWrapper(self, collate_fn)
