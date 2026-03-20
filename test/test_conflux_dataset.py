import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from codon.utils.dataset.conflux.base import ConfluxDataset

dataset = ConfluxDataset('./test_dataset').add_schema('input', 'text', 'txt')
dataset.set_compression_mode('uncompressed')

with dataset.open() as ds:
    key = ds.write({
        'input': 'text'
    })
    print(key)

print(dataset.manifest)