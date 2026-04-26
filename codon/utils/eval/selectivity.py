import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict
from .base import BaseAnalyzer, AnalysisResult

class NeuronSelectivity(BaseAnalyzer):
    '''
    Analyzer for computing and visualizing neuron class selectivity.
    It identifies neurons that respond most strongly to specific classes.
    Inherits from BaseAnalyzer.
    '''

    def __init__(
        self,
        class_info: Union[int, list[str]],
        lang: str = None
    ) -> None:
        '''
        Initializes the NeuronSelectivity analyzer.

        Args:
            class_info (Union[int, list[str]]): Information about classes (number or list of names).
            lang (str, optional): Language for visualization titles/labels ('en' or 'zh'). Defaults to None.
        '''
        super().__init__(class_info, lang=lang)
    
    @torch.no_grad()
    def analyse(
        self,
        feature_extractor: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        layer_name: str = None,
        name: str = '',
        top_k: int = 10,
        max_samples: int = 2000
    ) -> AnalysisResult:
        '''
        Analyzes the selectivity of neurons in a specified layer.

        Args:
            feature_extractor (torch.nn.Module): The feature extraction model.
            data_loader (torch.utils.data.DataLoader): The DataLoader providing input data.
            layer_name (str, optional): The specific layer to analyze. Automatically finds the first Linear/Conv2d if None.
            name (str, optional): Optional name to append to the plot title. Defaults to ''.
            top_k (int, optional): The number of top selective neurons to visualize. Defaults to 10.
            max_samples (int, optional): Maximum number of samples to process. Defaults to 2000.

        Returns:
            AnalysisResult: An object containing the generated plots and data regarding neuron selectivity.
        '''
        feature_extractor.eval()
        device = next(feature_extractor.parameters()).device
        
        if layer_name is None:
            for name, module in feature_extractor.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    layer_name = name
        
        activations = []
        targets_list = []
        
        def hook_fn(module, input, output):
            feat = output.detach()
            if feat.dim() > 2:
                feat = feat.mean(dim=[2, 3])
            activations.append(feat.cpu())
        
        target_module = dict(feature_extractor.named_modules())[layer_name]
        hook = target_module.register_forward_hook(hook_fn)
        
        sample_count = 0
        for inputs, targets in data_loader:
            if sample_count >= max_samples:
                break
            inputs = inputs.to(device)
            feature_extractor(inputs)
            targets_list.append(targets)
            sample_count += inputs.size(0)
        
        hook.remove()
        
        all_activations = torch.cat(activations, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        
        n_neurons = all_activations.shape[1]
        selectivity_indices = np.zeros(n_neurons)
        tuning_curves = np.zeros((n_neurons, self.class_num))
        
        for neuron_idx in range(n_neurons):
            neuron_acts = all_activations[:, neuron_idx].numpy()
            
            for class_idx in range(self.class_num):
                mask = (all_targets == class_idx).numpy()
                if mask.sum() > 0:
                    tuning_curves[neuron_idx, class_idx] = neuron_acts[mask].mean()
            
            mean_responses = tuning_curves[neuron_idx]
            if mean_responses.sum() > 0:
                max_response = mean_responses.max()
                mean_of_others = (mean_responses.sum() - max_response) / (self.class_num - 1)
                selectivity_indices[neuron_idx] = (max_response - mean_of_others) / (max_response + mean_of_others + 1e-8)
        
        top_indices = np.argsort(selectivity_indices)[-top_k:][::-1]
        
        def plot_fn(ax):
            n_plots = min(top_k, 6)
            
            for i, neuron_idx in enumerate(top_indices[:n_plots]):
                ax_sub = plt.subplot(2, 3, i + 1)
                
                responses = tuning_curves[neuron_idx]
                x_pos = np.arange(self.class_num)
                
                ax_sub.bar(x_pos, responses, alpha=0.7)
                ax_sub.set_title(f"{self.t['neuron']} {neuron_idx}\n"
                               f"{self.t['selectivity']}: {selectivity_indices[neuron_idx]:.3f}",
                               fontsize=9)
                
                if self.class_num <= 20:
                    ax_sub.set_xticks(x_pos)
                    ax_sub.set_xticklabels(self.class_name, rotation=45, ha='right', fontsize=7)
                else:
                    ax_sub.set_xlabel(self.t['class'])
                
                ax_sub.set_ylabel(self.t['mean_activation'], fontsize=8)
                ax_sub.grid(axis='y', alpha=0.3)
            
            title = self.t['selectivity_title']
            if name:
                title += f' - {name}'
            if layer_name:
                title += f' ({layer_name})'
            
            plt.suptitle(title, fontsize=12, y=0.98)
        
        fig = plt.figure(figsize=(15, 8))
        plot_fn(None)
        plt.tight_layout()
        
        return AnalysisResult(
            fig=fig, ax=fig.axes,
            data={
                'selectivity_indices': selectivity_indices,
                'tuning_curves': tuning_curves,
                'top_neurons': top_indices
            },
            plot_func=plot_fn
        )
