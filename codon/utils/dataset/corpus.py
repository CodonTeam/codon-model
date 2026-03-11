import csv
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable

import pandas as pd

from codon.utils.safecode import safecode

from .base import CodonDataset
from .flatdata import FlatDataset


class FileType(Enum):
    PARQUET = auto()
    JSONL = auto()
    CSV = auto()


@dataclass
class CorpusData:
    '''
    Represents a single corpus data entry with content, token count, and UUID.

    Attributes:
        content (str): The text content.
        num_token (int): The number of tokens (characters) in the content.
        uuid (str): Unique identifier for this entry.
    '''
    content: str
    num_token: int
    uuid: str


class CorpusDataset(CodonDataset):
    '''
    A dataset for managing linguistic corpora with token counting.

    This class maintains a folder of corpus files (PARQUET, JSONL, CSV) and
    tracks metadata in a JSON configuration file. Token count is calculated
    based on character count. Supports Key-Value access with both string keys
    (filename:row_number) and integer keys (global row index).

    Attributes:
        folder_path (str): Path to the folder containing corpus files.
        config_path (str): Path to the configuration JSON file.
        _config (dict): Configuration dictionary loaded from JSON.
        _total_token (int): Total token count across all files.
        _file_index (list): List of file metadata for index mapping.
        _cumulative_rows (list): Cumulative row counts for index mapping.
    '''

    def __init__(
        self,
        folder_path: str,
        file_type: FileType | None = None,
        file_limit: int = 2 * 1024 * 1024 * 1024,
        save_interval: int = 100
    ) -> None:
        '''
        Initializes the CorpusDataset.

        Args:
            folder_path (str): Path to the folder containing corpus files.
            file_type (FileType | None): The file type for storing data. If not provided,
                will attempt to read from config.json. Required for new datasets.
            file_limit (int): Maximum file size in bytes. Defaults to 2GB.
            save_interval (int): Number of additions before auto-saving config.
                Defaults to 100. Set to 1 to save after every addition.

        Raises:
            ValueError: If file_type is not provided and config.json does not exist.
        '''
        self.folder_path = folder_path
        self.config_path = os.path.join(folder_path, 'config.json')
        self.file_limit = file_limit
        self.save_interval = save_interval
        self._config: dict[str, Any] = {}
        self._total_token: int = 0
        self._file_index: list[dict[str, Any]] = []
        self._cumulative_rows: list[int] = [0]
        self._current_file_size: int = 0
        self._current_file_idx: int = 0
        self.file_type: FileType | None = file_type
        self._flat_dataset_cache: dict[str, FlatDataset] = {}
        self._parquet_buffer: list[dict[str, Any]] = []
        self._file_name_to_idx: dict[str, int] = {}
        self._lock = threading.Lock()
        self._pending_changes: bool = False
        self._add_count: int = 0

        # Create folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        # Load or initialize config
        self._load_config()

        # Determine file_type
        if self.file_type is None:
            if 'file_type' in self._config:
                self.file_type = FileType[self._config['file_type']]
            else:
                raise ValueError(
                    'file_type must be specified when creating a new CorpusDataset'
                )

        # Store file_type in config
        self._config['file_type'] = self.file_type.name
        self._save_config()

    def __enter__(self) -> 'CorpusDataset':
        '''
        Enters the context manager.

        Returns:
            CorpusDataset: The dataset instance.
        '''
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        '''
        Exits the context manager, ensuring all data is flushed.
        '''
        self.close()

    def close(self) -> None:
        '''
        Closes the dataset, flushing all buffers and saving config.

        This method should be called when finished using the dataset to ensure
        all buffered data is written to disk and configuration is saved.
        '''
        with self._lock:
            # Flush parquet buffer if not empty
            if self._parquet_buffer:
                current_filename = f'corpus_{self._current_file_idx:03d}.{self._get_file_extension()}'
                current_file_path = os.path.join(self.folder_path, current_filename)
                self._flush_parquet_buffer(current_file_path)

            # Save config if there are pending changes
            if self._pending_changes:
                self._save_config()
                self._rebuild_index()
                self._pending_changes = False

    def _load_config(self) -> None:
        '''
        Loads configuration from JSON file or creates a new one.
        '''
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        else:
            self._config = {
                'version': '1.0',
                'total_token': 0,
                'files': []
            }
            self._save_config()

        # Rebuild file index and cumulative rows
        self._rebuild_index()

    def _save_config(self) -> None:
        '''
        Saves configuration to JSON file.
        '''
        self._config['total_token'] = self._total_token
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

    def _rebuild_index(self) -> None:
        '''
        Rebuilds the file index and cumulative row counts.
        '''
        self._file_index = []
        self._cumulative_rows = [0]
        self._total_token = 0
        self._file_name_to_idx = {}

        for file_info in self._config.get('files', []):
            self._file_index.append(file_info)
            self._file_name_to_idx[file_info['filename']] = len(self._file_index) - 1
            self._total_token += file_info.get('num_token', 0)
            self._cumulative_rows.append(
                self._cumulative_rows[-1] + file_info.get('num_rows', 0)
            )

    def _detect_file_type(self, file_path: str) -> FileType:
        '''
        Detects file type based on extension.

        Args:
            file_path (str): Path to the file.

        Returns:
            FileType: The detected file type.

        Raises:
            ValueError: If file type is not supported.
        '''
        if file_path.endswith('.parquet'):
            return FileType.PARQUET
        elif file_path.endswith('.jsonl'):
            return FileType.JSONL
        elif file_path.endswith('.csv'):
            return FileType.CSV
        else:
            raise ValueError(f'Unsupported file type: {file_path}')

    def _count_tokens(self, content: str) -> int:
        '''
        Counts tokens in content based on character count.

        Args:
            content (str): The text content.

        Returns:
            int: The number of tokens (characters).
        '''
        return len(content)

    def _load_file_data(self, file_path: str, file_type: FileType) -> list[dict[str, Any]]:
        '''
        Loads data from a file.

        Args:
            file_path (str): Path to the file.
            file_type (FileType): Type of the file.

        Returns:
            list[dict[str, Any]]: List of dictionaries representing rows.

        Raises:
            ValueError: If file type is not supported.
        '''
        if file_type == FileType.PARQUET:
            df = pd.read_parquet(file_path)
            return df.to_dict('records')
        elif file_type == FileType.JSONL:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        elif file_type == FileType.CSV:
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        else:
            raise ValueError(f'Unsupported file type: {file_type}')

    def add(self, data: str) -> None:
        '''
        Adds a string data entry to the dataset. If adding the data would exceed
        the file size limit, automatically creates a new file.

        Args:
            data (str): The text content to add.

        Raises:
            TypeError: If data is not a string.
            ValueError: If data is an empty string.
        '''
        # Input validation
        if not isinstance(data, str):
            raise TypeError(f'data must be str, got {type(data).__name__}')
        if not data:
            raise ValueError('data cannot be empty string')

        with self._lock:
            # Calculate token count (character count)
            num_token = self._count_tokens(data)
            data_size = len(data.encode('utf-8'))

            # Check if adding this data would exceed limit
            if self._current_file_size + data_size > self.file_limit and len(self._config['files']) > 0:
                # Flush parquet buffer before starting new file
                if self.file_type == FileType.PARQUET and self._parquet_buffer:
                    current_filename = f'corpus_{self._current_file_idx:03d}.{self._get_file_extension()}'
                    current_file_path = os.path.join(self.folder_path, current_filename)
                    self._flush_parquet_buffer(current_file_path)
                # Start a new file
                self._current_file_idx += 1
                self._current_file_size = 0

            # Get or create current file
            current_filename = f'corpus_{self._current_file_idx:03d}.{self._get_file_extension()}'
            current_file_path = os.path.join(self.folder_path, current_filename)

            # Generate UUID for this entry with timestamp prefix
            entry_uuid = f'{int(time.time()):x}_{safecode(12)}'

            # Prepare row data
            row_data = {'content': data, 'uuid': entry_uuid}

            # Append to file
            if self.file_type == FileType.JSONL:
                with open(current_file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(row_data, ensure_ascii=False) + '\n')
            elif self.file_type == FileType.CSV:
                file_exists = os.path.exists(current_file_path)
                with open(current_file_path, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['content', 'uuid'])
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row_data)
            elif self.file_type == FileType.PARQUET:
                # For parquet, accumulate in memory and write periodically
                self._parquet_buffer.append(row_data)
                # Write to parquet every 1000 rows or when buffer is large
                if len(self._parquet_buffer) >= 1000:
                    self._flush_parquet_buffer(current_file_path)

            # Update file info
            self._update_file_info(current_filename, num_token, data_size)

    def _get_file_extension(self) -> str:
        '''
        Gets the file extension based on file_type.

        Returns:
            str: The file extension without the dot.

        Raises:
            ValueError: If file type is not supported.
        '''
        if self.file_type == FileType.PARQUET:
            return 'parquet'
        elif self.file_type == FileType.JSONL:
            return 'jsonl'
        elif self.file_type == FileType.CSV:
            return 'csv'
        else:
            raise ValueError(f'Unsupported file type: {self.file_type}')

    def _flush_parquet_buffer(self, file_path: str) -> None:
        '''
        Flushes accumulated parquet data to file.

        Args:
            file_path (str): Path to the parquet file.
        '''
        if not self._parquet_buffer:
            return

        df = pd.DataFrame(self._parquet_buffer)
        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_parquet(file_path, index=False)
        self._parquet_buffer = []

    def _update_file_info(self, filename: str, num_token: int, data_size: int) -> None:
        '''
        Updates file information in config with delayed saving.

        Args:
            filename (str): The filename.
            num_token (int): Number of tokens added.
            data_size (int): Size of data in bytes.
        '''
        # Find or create file info
        file_info = None
        for info in self._config['files']:
            if info['filename'] == filename:
                file_info = info
                break

        if file_info is None:
            file_info = {
                'filename': filename,
                'file_type': self.file_type.name,
                'num_rows': 0,
                'num_token': 0,
                'created_at': datetime.now().isoformat()
            }
            self._config['files'].append(file_info)

        file_info['num_rows'] += 1
        file_info['num_token'] += num_token
        self._total_token += num_token
        self._current_file_size += data_size

        # Increment add count and mark pending changes
        self._add_count += 1
        self._pending_changes = True

        # Save config periodically based on save_interval
        if self._add_count % self.save_interval == 0:
            self._save_config()
            self._rebuild_index()
            self._pending_changes = False

    def flush(self) -> None:
        '''
        Manually flushes all buffers and saves configuration.

        This method forces a write of any buffered parquet data and saves
        the configuration file, regardless of the save_interval setting.
        '''
        with self._lock:
            # Flush parquet buffer
            if self._parquet_buffer:
                current_filename = f'corpus_{self._current_file_idx:03d}.{self._get_file_extension()}'
                current_file_path = os.path.join(self.folder_path, current_filename)
                self._flush_parquet_buffer(current_file_path)

            # Save config if pending changes
            if self._pending_changes:
                self._save_config()
                self._rebuild_index()
                self._pending_changes = False

    def add_from_file(
        self,
        file_path: str,
        fields: list[str],
        separator: str = '',
        skip_empty: bool = True,
        strict_fields: bool = False,
        progress_callback: Callable[[int, int], None] | None = None
    ) -> int:
        '''
        Adds data from a file by reading specified fields using streaming.

        This method uses FlatDataset for memory-efficient streaming read of
        large files, avoiding loading the entire file into memory.

        Args:
            file_path (str): Path to the source file.
            fields (list[str]): List of field names to read and concatenate.
                If a field doesn't exist and strict_fields is False, it is skipped.
            separator (str): Separator string between concatenated fields.
                Defaults to '' (no separator).
            skip_empty (bool): Whether to skip rows with empty content after
                field extraction. Defaults to True.
            strict_fields (bool): If True, raises ValueError when a specified
                field is missing from a row. Defaults to False.
            progress_callback (Callable[[int, int], None] | None): Optional
                callback function called with (current_row, total_rows) for
                progress monitoring. Defaults to None.

        Returns:
            int: Number of rows successfully added to the dataset.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If strict_fields is True and a field is missing.
        '''
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found: {file_path}')

        # Use FlatDataset for memory-efficient streaming
        flat_dataset = FlatDataset(file_path, in_memory=False, shuffle=False)
        total_rows = len(flat_dataset)
        rows_added = 0

        for idx in range(total_rows):
            row = flat_dataset.get_value(idx)

            # Extract and concatenate specified fields
            content_parts = []
            for field in fields:
                if field in row:
                    value = row[field]
                    if value is not None:
                        content_parts.append(str(value))
                elif strict_fields:
                    raise ValueError(
                        f"Field '{field}' not found in row {idx}. "
                        f"Available fields: {list(row.keys())}"
                    )

            # Skip empty content if configured
            if not content_parts:
                if skip_empty:
                    continue
                else:
                    content_parts = ['']

            concatenated_content = separator.join(content_parts)

            # Skip if empty after concatenation and skip_empty is True
            if skip_empty and not concatenated_content:
                continue

            self.add(concatenated_content)
            rows_added += 1

            # Call progress callback if provided
            if progress_callback is not None:
                progress_callback(idx + 1, total_rows)

        return rows_added

    def _get_or_create_flat_dataset(self, file_path: str) -> FlatDataset:
        '''
        Gets or creates a FlatDataset instance for lazy loading.

        This method caches FlatDataset instances to avoid recreating them
        for repeated access to the same file.

        Args:
            file_path (str): Path to the corpus file.

        Returns:
            FlatDataset: A FlatDataset instance for lazy loading the file.
        '''
        if file_path not in self._flat_dataset_cache:
            self._flat_dataset_cache[file_path] = FlatDataset(
                file_path,
                in_memory=False,
                shuffle=False
            )
        return self._flat_dataset_cache[file_path]

    def get(self, key: int | str) -> CorpusData:
        '''
        Retrieves a corpus data entry by key using lazy loading.

        Args:
            key (int | str): The key to retrieve. Can be:
                - int: Global row index (0, 1, 2, ...)
                - str: "filename:row_number" format (e.g., "corpus_001.parquet:5")

        Returns:
            CorpusData: The corpus data at the specified key.

        Raises:
            IndexError: If index is out of range.
            KeyError: If string key format is invalid.
        '''
        with self._lock:
            # Handle string key format: "filename:row_number"
            if isinstance(key, str):
                if ':' not in key:
                    raise KeyError(f'Invalid key format: {key}. Expected "filename:row_number"')
                filename, row_str = key.rsplit(':', 1)
                try:
                    row_in_file = int(row_str)
                except ValueError:
                    raise KeyError(f'Invalid row number in key: {key}')

                # Find file by filename using optimized lookup
                if filename not in self._file_name_to_idx:
                    raise KeyError(f'File not found: {filename}')

                file_idx = self._file_name_to_idx[filename]
                file_info = self._file_index[file_idx]

                if row_in_file < 0 or row_in_file >= file_info['num_rows']:
                    raise IndexError(f'Row {row_in_file} out of range for {filename}')
            else:
                # Handle integer key: global row index
                if key < 0 or key >= self._cumulative_rows[-1]:
                    raise IndexError(f'Index {key} out of range')

                # Find which file contains this index
                file_idx = 0
                for i, cumulative_row in enumerate(self._cumulative_rows[1:], 1):
                    if key < cumulative_row:
                        file_idx = i - 1
                        break

                row_in_file = key - self._cumulative_rows[file_idx]
                file_info = self._file_index[file_idx]

            # Use FlatDataset for lazy loading
            file_path = os.path.join(self.folder_path, file_info['filename'])
            flat_dataset = self._get_or_create_flat_dataset(file_path)
            row = flat_dataset.get_value(row_in_file)

            # Find content column (try common names)
            content_column = None
            for col_name in ['content', 'text', 'data', 'corpus']:
                if col_name in row:
                    content_column = col_name
                    break

            if content_column is None:
                # Use first column if no standard name found
                content_column = list(row.keys())[0]

            content = str(row[content_column])
            num_token = self._count_tokens(content)
            uuid = str(row.get('uuid', ''))

            return CorpusData(content=content, num_token=num_token, uuid=uuid)

    def __len__(self) -> int:
        '''
        Returns the total number of entries in the dataset.

        Returns:
            int: The total number of entries.
        '''
        return self._cumulative_rows[-1] if self._cumulative_rows else 0

    def __getitem__(self, key: int | str) -> CorpusData:
        '''
        Retrieves an entry by key using bracket notation.

        Args:
            key (int | str): The key to retrieve (int index or str "filename:row").

        Returns:
            CorpusData: The corpus data at the specified key.
        '''
        return self.get(key)

    def __repr__(self) -> str:
        '''
        Returns a string representation of the dataset for debugging.

        Returns:
            str: A string representation showing key dataset information.
        '''
        return (
            f'CorpusDataset('
            f'folder={self.folder_path!r}, '
            f'files={len(self._file_index)}, '
            f'rows={len(self)}, '
            f'tokens={self._total_token}, '
            f'type={self.file_type.name if self.file_type else None})'
        )
