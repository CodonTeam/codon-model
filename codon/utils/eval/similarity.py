import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union
from .base import BaseAnalyzer, AnalysisResult

class CrossCosineSimilarityMap(BaseAnalyzer):
    def __init__(self, class_info: Union[int, list[str]] = None, lang: str = None):
        super().__init__(class_info, lang=lang)
    
    def analyse(self, matrix: torch.Tensor, name: str = '', mask_upper: bool = False) -> Union[AnalysisResult, None]:
        if matrix.dim() != 2: return None

        if self.class_num == 0:
            self.class_num = matrix.shape[0]
            self.class_name = [str(i) for i in range(self.class_num)]

        matrix_normalized = torch.nn.functional.normalize(matrix, p=2, dim=1)
        C_matrix = (matrix_normalized @ matrix_normalized.T).detach().cpu().numpy()

        def plot_fn(ax):
            mask = np.triu(np.ones_like(C_matrix, dtype=bool)) if mask_upper else None
            
            sns.heatmap(
                C_matrix, fmt='.2f', cmap='coolwarm',
                annot=self.show_annot,
                xticklabels=self.get_ticklabels(),
                yticklabels=self.get_ticklabels(),
                ax=ax, mask=mask
            )
            
            title = self.t['similarity_title']
            if name: title += f' - {name}'
                
            ax.set_title(title)

        fig, ax = plt.subplots(figsize=self._get_figsize())
        plot_fn(ax)

        return AnalysisResult(fig=fig, ax=ax, data=C_matrix, plot_func=plot_fn)