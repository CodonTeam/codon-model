import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Union
from .base import BaseAnalyzer, AnalysisResult

class TSNEMap(BaseAnalyzer):
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
        max_samples: int = 2000,
        perplexity: float = 30.0
    ) -> AnalysisResult:
        feature_extractor.eval()
        device = next(feature_extractor.parameters()).device
        
        all_features = []
        all_targets = []
        
        for inputs, targets in self.data_loader:
            inputs = inputs.to(device)
            features = feature_extractor(inputs)
            
            if features.dim() > 2: 
                features = features.view(features.size(0), -1)
                
            all_features.append(features.cpu())
            all_targets.append(targets.cpu())
            
        features_tensor = torch.cat(all_features, dim=0).numpy()
        targets_tensor = torch.cat(all_targets, dim=0).numpy()
        
        if len(features_tensor) > max_samples:
            indices = np.random.choice(len(features_tensor), max_samples, replace=False)
            features_tensor = features_tensor[indices]
            targets_tensor = targets_tensor[indices]

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embedded = tsne.fit_transform(features_tensor)

        def plot_fn(ax):
            target_names = [self.class_name[i] for i in targets_tensor]
            
            palette = 'tab10' if self.class_num <= 10 else 'husl'
            
            sns.scatterplot(
                x=embedded[:, 0], y=embedded[:, 1],
                hue=target_names,
                hue_order=self.class_name,
                palette=palette,
                legend='full',
                alpha=0.7,
                s=30,
                ax=ax
            )
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            title = self.t['tsne_title']
            if name: title += f' - {name}'
            ax.set_title(title)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        base_w, base_h = self._get_figsize()
        fig, ax = plt.subplots(figsize=(base_w + 2, base_h))
        
        plot_fn(ax)
        
        return AnalysisResult(fig=fig, ax=ax, data=embedded, plot_func=plot_fn)