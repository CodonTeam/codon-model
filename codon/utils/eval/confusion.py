import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
from sklearn.metrics import confusion_matrix

class ConfusionMap:
    def __init__(
        self,
        class_info: Union[int, list[str]],
        val_loader: torch.utils.data.DataLoader,
        path: str
    ):
        if isinstance(class_info, int):
            self.class_num = class_info
            self.class_name = [str(i) for i in range(class_info)]
        else:
            self.class_num = len(class_info)
            self.class_name = class_info
        
        if not os.path.isdir(path): os.makedirs(path)
            
        self.path = path
        self.data_loader = val_loader
    
    @torch.no_grad()
    def analyse(self, model: torch.nn.Module, name: str, show: bool = False) -> str:
        model.eval()
        device = next(model.parameters()).device
        
        all_preds = []
        all_targets = []
        
        for inputs, targets in self.data_loader:
            inputs: torch.Tensor = inputs.to(device)
            outputs: torch.Tensor = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
        cm = confusion_matrix(all_targets, all_preds, labels=range(self.class_num))
        
        plt.figure(figsize=(max(8, self.class_num * 0.8), max(6, self.class_num * 0.6)))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=self.class_name, 
            yticklabels=self.class_name
        )
        
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        save_path = os.path.join(self.path, f"{name}_confusion.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return save_path