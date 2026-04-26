import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict
from .base import BaseAnalyzer, AnalysisResult

class ActivationDistribution(BaseAnalyzer):
    def __init__(
        self,
        class_info: Union[int, list[str]] = None,
        lang: str = None
    ):
        super().__init__(class_info, lang=lang)
    
    @torch.no_grad()
    def analyse(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        layer_names: list[str] = None,
        name: str = '',
        max_batches: int = 10
    ) -> AnalysisResult:
        model.eval()
        device = next(model.parameters()).device
        
        if layer_names is None:
            layer_names = [name for name, _ in model.named_modules() 
                          if isinstance(_, (torch.nn.Conv2d, torch.nn.Linear))]
        
        activations: Dict[str, list] = {layer: [] for layer in layer_names}
        
        hooks = []
        def get_activation(name):
            def hook(module, input, output):
                activations[name].append(output.detach().cpu().flatten())
            return hook
        
        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        batch_count = 0
        for inputs, _ in data_loader:
            if batch_count >= max_batches:
                break
            inputs = inputs.to(device)
            model(inputs)
            batch_count += 1
        
        for hook in hooks:
            hook.remove()
        
        stats = {}
        for layer in layer_names:
            if activations[layer]:
                acts = torch.cat(activations[layer]).numpy()
                stats[layer] = {
                    'mean': np.mean(acts),
                    'std': np.std(acts),
                    'sparsity': np.mean(acts == 0),
                    'data': acts
                }
        
        def plot_fn(ax):
            n_layers = len(stats)
            if n_layers == 0:
                return
            
            positions = []
            data_to_plot = []
            labels = []
            
            for i, (layer, stat) in enumerate(stats.items()):
                sample_size = min(10000, len(stat['data']))
                sampled = np.random.choice(stat['data'], sample_size, replace=False)
                data_to_plot.append(sampled)
                positions.append(i)
                
                short_name = layer.split('.')[-1] if '.' in layer else layer
                labels.append(f"{short_name}\n"
                            f"{self.t['mean']}: {stat['mean']:.3f}\n"
                            f"{self.t['sparsity']}: {stat['sparsity']:.2%}")
            
            parts = ax.violinplot(data_to_plot, positions=positions, 
                                 showmeans=True, showmedians=True)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(self.t['activation_value'])
            ax.grid(axis='y', alpha=0.3)
            
            title = self.t['activation_title']
            if name:
                title += f' - {name}'
            ax.set_title(title)
        
        fig, ax = plt.subplots(figsize=(max(12, len(stats) * 2), 6))
        plot_fn(ax)
        plt.tight_layout()
        
        return AnalysisResult(fig=fig, ax=ax, data=stats, plot_func=plot_fn)