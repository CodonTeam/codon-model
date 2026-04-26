import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Union
from .base import BaseAnalyzer, AnalysisResult

class ConfusionMap(BaseAnalyzer):
    def __init__(
        self,
        class_info: Union[int, list[str]],
        val_loader: torch.utils.data.DataLoader,
        lang: str = None
    ):
        super().__init__(class_info, lang=lang)
        self.data_loader = val_loader
    
    @torch.no_grad()
    def analyse(self, model: torch.nn.Module, name: str = '') -> AnalysisResult:
        model.eval()
        device = next(model.parameters()).device
        
        all_preds = []
        all_targets = []
        
        for inputs, targets in self.data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
        cm = confusion_matrix(all_targets, all_preds, labels=range(self.class_num))
        
        def plot_fn(ax):
            sns.heatmap(
                cm, fmt='d', cmap='Blues', ax=ax,
                annot=self.show_annot,
                xticklabels=self.get_ticklabels(), 
                yticklabels=self.get_ticklabels(),
            )
            
            title = self.t['confusion_title']
            if name: title += f' - {name}'
            
            ax.set_title(title)
            ax.set_ylabel(self.t['actual'])
            ax.set_xlabel(self.t['predicted'])

        fig, ax = plt.subplots(figsize=self._get_figsize())
        plot_fn(ax)
        
        return AnalysisResult(fig=fig, ax=ax, data=cm, plot_func=plot_fn)