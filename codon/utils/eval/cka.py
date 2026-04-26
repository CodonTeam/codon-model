import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union
from .base import BaseAnalyzer, AnalysisResult

class CKAMap(BaseAnalyzer):
    '''
    Analyzer for computing and visualizing the Centered Kernel Alignment (CKA) similarity
    between representations of two neural network models.
    Inherits from BaseAnalyzer.
    '''

    def __init__(
        self,
        class_info: Union[int, list[str]] = None,
        lang: str = None
    ) -> None:
        '''
        Initializes the CKAMap analyzer.

        Args:
            class_info (Union[int, list[str]], optional): Information about classes. Defaults to None.
            lang (str, optional): Language for visualization titles/labels ('en' or 'zh'). Defaults to None.
        '''
        super().__init__(class_info, lang=lang)
    
    @staticmethod
    def _centering(K: torch.Tensor) -> torch.Tensor:
        '''
        Centers the kernel matrix K.

        Args:
            K (torch.Tensor): The kernel matrix to center.

        Returns:
            torch.Tensor: The centered kernel matrix.
        '''
        n = K.shape[0]
        unit = torch.ones([n, n], device=K.device)
        I = torch.eye(n, device=K.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)
    
    @staticmethod
    def _linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
        '''
        Computes the linear CKA similarity between two sets of features.

        Args:
            X (torch.Tensor): Features from the first model/layer.
            Y (torch.Tensor): Features from the second model/layer.

        Returns:
            float: The linear CKA similarity score.
        '''
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        XY = torch.matmul(X.T, Y)
        hsic = torch.norm(XY, p='fro') ** 2
        
        XX = torch.matmul(X.T, X)
        YY = torch.matmul(Y.T, Y)
        
        var_X = torch.norm(XX, p='fro') ** 2
        var_Y = torch.norm(YY, p='fro') ** 2
        
        return (hsic / torch.sqrt(var_X * var_Y)).item()
    
    @torch.no_grad()
    def analyse(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        layer_names_a: list[str],
        layer_names_b: list[str] = None,
        name: str = '',
        max_samples: int = 1000
    ) -> AnalysisResult:
        '''
        Analyzes the CKA similarity between specified layers of two models.

        Args:
            model_a (torch.nn.Module): The first PyTorch model.
            model_b (torch.nn.Module): The second PyTorch model.
            data_loader (torch.utils.data.DataLoader): The DataLoader providing input data.
            layer_names_a (list[str]): List of layer names to analyze from model_a.
            layer_names_b (list[str], optional): List of layer names to analyze from model_b. Defaults to layer_names_a if None.
            name (str, optional): Optional name to append to the plot title. Defaults to ''.
            max_samples (int, optional): Maximum number of samples to process. Defaults to 1000.

        Returns:
            AnalysisResult: An object containing the generated heatmap and the CKA similarity matrix.
        '''
        if layer_names_b is None:
            layer_names_b = layer_names_a
        
        model_a.eval()
        model_b.eval()
        device_a = next(model_a.parameters()).device
        device_b = next(model_b.parameters()).device
        
        features_a = {layer: [] for layer in layer_names_a}
        features_b = {layer: [] for layer in layer_names_b}
        
        def get_activation(features_dict, name):
            def hook(module, input, output):
                feat = output.detach()
                if feat.dim() > 2:
                    feat = feat.view(feat.size(0), -1)
                features_dict[name].append(feat.cpu())
            return hook
        
        hooks_a = []
        for name, module in model_a.named_modules():
            if name in layer_names_a:
                hooks_a.append(module.register_forward_hook(
                    get_activation(features_a, name)))
        
        hooks_b = []
        for name, module in model_b.named_modules():
            if name in layer_names_b:
                hooks_b.append(module.register_forward_hook(
                    get_activation(features_b, name)))
        
        sample_count = 0
        for inputs, _ in data_loader:
            if sample_count >= max_samples:
                break
            
            model_a(inputs.to(device_a))
            model_b(inputs.to(device_b))
            
            sample_count += inputs.size(0)
        
        for hook in hooks_a + hooks_b:
            hook.remove()
        
        for layer in layer_names_a:
            if features_a[layer]:
                features_a[layer] = torch.cat(features_a[layer], dim=0)
        
        for layer in layer_names_b:
            if features_b[layer]:
                features_b[layer] = torch.cat(features_b[layer], dim=0)
        
        n_a = len(layer_names_a)
        n_b = len(layer_names_b)
        cka_matrix = np.zeros((n_a, n_b))
        
        for i, layer_a in enumerate(layer_names_a):
            for j, layer_b in enumerate(layer_names_b):
                if layer_a in features_a and layer_b in features_b:
                    X = features_a[layer_a]
                    Y = features_b[layer_b]
                    cka_matrix[i, j] = self._linear_CKA(X, Y)
        
        def plot_fn(ax):
            short_labels_a = [name.split('.')[-1] if '.' in name else name 
                            for name in layer_names_a]
            short_labels_b = [name.split('.')[-1] if '.' in name else name 
                            for name in layer_names_b]
            
            sns.heatmap(
                cka_matrix,
                annot=True if max(n_a, n_b) <= 15 else False,
                fmt='.2f',
                cmap='viridis',
                xticklabels=short_labels_b,
                yticklabels=short_labels_a,
                ax=ax,
                vmin=0,
                vmax=1
            )
            
            title = self.t['cka_title']
            if name:
                title += f' - {name}'
            ax.set_title(title)
            
            ax.set_xlabel(self.t['model_b'])
            ax.set_ylabel(self.t['model_a'])
        
        fig, ax = plt.subplots(figsize=(max(8, n_b * 0.6), max(6, n_a * 0.5)))
        plot_fn(ax)
        plt.tight_layout()
        
        return AnalysisResult(fig=fig, ax=ax, data=cka_matrix, plot_func=plot_fn)
