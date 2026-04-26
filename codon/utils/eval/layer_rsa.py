import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, Dict
from .base import BaseAnalyzer, AnalysisResult

class LayerRSAMap(BaseAnalyzer):
    '''
    Analyzer for computing and visualizing the Representational Similarity Analysis (RSA)
    across different layers of a single neural network model.
    Inherits from BaseAnalyzer.
    '''

    def __init__(
        self,
        class_info: Union[int, list[str]] = None,
        lang: str = None
    ) -> None:
        '''
        Initializes the LayerRSAMap analyzer.

        Args:
            class_info (Union[int, list[str]], optional): Information about classes. Defaults to None.
            lang (str, optional): Language for visualization titles/labels ('en' or 'zh'). Defaults to None.
        '''
        super().__init__(class_info, lang=lang)
    
    @torch.no_grad()
    def analyse(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        layer_names: list[str],
        name: str = '',
        max_samples: int = 1000
    ) -> AnalysisResult:
        '''
        Analyzes the representational similarity between specified layers of the model.

        Args:
            model (torch.nn.Module): The PyTorch model to evaluate.
            data_loader (torch.utils.data.DataLoader): The DataLoader providing input data.
            layer_names (list[str]): List of layer names to analyze and compare.
            name (str, optional): Optional name to append to the plot title. Defaults to ''.
            max_samples (int, optional): Maximum number of samples to process. Defaults to 1000.

        Returns:
            AnalysisResult: An object containing the generated heatmap and the RSA similarity matrix.
        '''
        model.eval()
        device = next(model.parameters()).device
        
        layer_features: Dict[str, list] = {layer: [] for layer in layer_names}
        
        hooks = []
        def get_activation(name):
            def hook(module, input, output):
                feat = output.detach()
                if feat.dim() > 2:
                    feat = feat.view(feat.size(0), -1)
                layer_features[name].append(feat.cpu())
            return hook
        
        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        sample_count = 0
        for inputs, _ in data_loader:
            if sample_count >= max_samples:
                break
            inputs = inputs.to(device)
            model(inputs)
            sample_count += inputs.size(0)
        
        for hook in hooks:
            hook.remove()
        
        centroids = {}
        for layer in layer_names:
            if layer_features[layer]:
                features = torch.cat(layer_features[layer], dim=0)
                centroids[layer] = features.mean(dim=0)
        
        n_layers = len(centroids)
        rsa_matrix = np.zeros((n_layers, n_layers))
        
        layer_list = list(centroids.keys())
        for i, layer_i in enumerate(layer_list):
            for j, layer_j in enumerate(layer_list):
                feat_i = centroids[layer_i]
                feat_j = centroids[layer_j]
                
                feat_i_norm = feat_i / (feat_i.norm() + 1e-8)
                feat_j_norm = feat_j / (feat_j.norm() + 1e-8)
                
                similarity = (feat_i_norm * feat_j_norm).sum().item()
                rsa_matrix[i, j] = similarity
        
        def plot_fn(ax):
            short_labels = [name.split('.')[-1] if '.' in name else name 
                          for name in layer_list]
            
            sns.heatmap(
                rsa_matrix,
                annot=True if n_layers <= 15 else False,
                fmt='.2f',
                cmap='viridis',
                xticklabels=short_labels,
                yticklabels=short_labels,
                ax=ax,
                vmin=-1,
                vmax=1,
                square=True
            )
            
            title = self.t['layer_rsa_title']
            if name:
                title += f' - {name}'
            ax.set_title(title)
            
            ax.set_xlabel(self.t['layer'])
            ax.set_ylabel(self.t['layer'])
        
        fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.6), max(6, n_layers * 0.5)))
        plot_fn(ax)
        plt.tight_layout()
        
        return AnalysisResult(fig=fig, ax=ax, data=rsa_matrix, plot_func=plot_fn)
