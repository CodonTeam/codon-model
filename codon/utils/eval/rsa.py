import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union
from .base import BaseAnalyzer, AnalysisResult

class RSAMap(BaseAnalyzer):
    def __init__(
        self,
        class_info: Union[int, list[str]],
        val_loader: torch.utils.data.DataLoader,
        lang: str = None
    ):
        super().__init__(class_info, lang=lang)
        self.data_loader = val_loader

    @torch.no_grad()
    def analyse(
        self, 
        feature_extractor: torch.nn.Module, 
        name: str = '',
        mask_upper: bool = False
    ) -> AnalysisResult:
        feature_extractor.eval()
        device = next(feature_extractor.parameters()).device
        
        class_features = {i: [] for i in range(self.class_num)}
        
        for inputs, targets in self.data_loader:
            inputs = inputs.to(device)
            features = feature_extractor(inputs)
            
            if features.dim() > 2: 
                features = features.view(features.size(0), -1)
                
            features = features.cpu()
            targets = targets.cpu()
            
            for feat, label in zip(features, targets):
                class_features[label.item()].append(feat)
                
        centroids = []
        for i in range(self.class_num):
            if len(class_features[i]) > 0:
                centroid = torch.stack(class_features[i]).mean(dim=0)
            else:
                centroid = torch.zeros_like(features[0])
            centroids.append(centroid)
            
        centroids_tensor = torch.stack(centroids) # Shape: (C, D)
        centered = centroids_tensor - centroids_tensor.mean(dim=1, keepdim=True)

        norms = centered.norm(p=2, dim=1, keepdim=True)
        norms[norms == 0] = 1e-8
        normalized = centered / norms
        
        rsa_matrix = (normalized @ normalized.T).numpy()

        def plot_fn(ax):
            mask = np.triu(np.ones_like(rsa_matrix, dtype=bool)) if mask_upper else None
            
            sns.heatmap(
                rsa_matrix, fmt='.2f', cmap='viridis',
                annot=self.show_annot,
                xticklabels=self.get_ticklabels(),
                yticklabels=self.get_ticklabels(),
                ax=ax, mask=mask, vmin=-1, vmax=1
            )
            
            title = self.t['rsa_title']
            if name: title += f' - {name}'
            ax.set_title(title)

        fig, ax = plt.subplots(figsize=self._get_figsize())
        plot_fn(ax)
        
        return AnalysisResult(fig=fig, ax=ax, data=rsa_matrix, plot_func=plot_fn)