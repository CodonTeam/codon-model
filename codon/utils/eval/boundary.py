import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from .base import BaseAnalyzer, AnalysisResult

class DecisionBoundaryMap(BaseAnalyzer):
    '''
    Analyzer for visualizing the decision boundaries and prediction confidence of a model
    using t-SNE embeddings of the features.
    Inherits from BaseAnalyzer.
    '''

    def __init__(
        self,
        class_info: Union[int, list[str]],
        lang: str = None
    ) -> None:
        '''
        Initializes the DecisionBoundaryMap analyzer.

        Args:
            class_info (Union[int, list[str]]): Information about classes (number or list of names).
            lang (str, optional): Language for visualization titles/labels ('en' or 'zh'). Defaults to None.
        '''
        super().__init__(class_info, lang=lang)
    
    @torch.no_grad()
    def analyse(
        self,
        model: torch.nn.Module,
        feature_extractor: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        tsne_embedded: np.ndarray = None,
        name: str = '',
        resolution: int = 100,
        max_samples: int = 2000
    ) -> AnalysisResult:
        '''
        Analyzes and plots the decision boundaries and confidence map.

        Args:
            model (torch.nn.Module): The classification model containing a fully connected layer (e.g., 'fc').
            feature_extractor (torch.nn.Module): The feature extraction part of the model.
            data_loader (torch.utils.data.DataLoader): The DataLoader providing input data.
            tsne_embedded (np.ndarray, optional): Pre-computed t-SNE embeddings. Defaults to None.
            name (str, optional): Optional name to append to the plot title. Defaults to ''.
            resolution (int, optional): Resolution of the grid used for boundary drawing. Defaults to 100.
            max_samples (int, optional): Maximum number of samples to use. Defaults to 2000.

        Returns:
            AnalysisResult: An object containing the generated plot and grid prediction/confidence data.
        '''
        model.eval()
        feature_extractor.eval()
        device = next(model.parameters()).device
        
        all_features = []
        all_targets = []
        all_inputs = []
        
        for inputs, targets in data_loader:
            inputs_device = inputs.to(device)
            features = feature_extractor(inputs_device)
            
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            
            all_features.append(features.cpu())
            all_targets.append(targets.cpu())
            all_inputs.append(inputs.cpu())
            
            if sum(f.size(0) for f in all_features) >= max_samples:
                break
        
        features_tensor = torch.cat(all_features, dim=0)[:max_samples]
        targets_tensor = torch.cat(all_targets, dim=0)[:max_samples]
        inputs_tensor = torch.cat(all_inputs, dim=0)[:max_samples]
        
        if tsne_embedded is None:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            embedded = tsne.fit_transform(features_tensor.numpy())
        else:
            embedded = tsne_embedded
        
        x_min, x_max = embedded[:, 0].min() - 1, embedded[:, 0].max() + 1
        y_min, y_max = embedded[:, 1].min() - 1, embedded[:, 1].max() + 1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        from scipy.spatial import cKDTree
        tree = cKDTree(embedded)
        distances, indices = tree.query(grid_points, k=5)
        
        weights = 1 / (distances + 1e-8)
        weights = weights / weights.sum(axis=1, keepdim=True)
        
        neighbor_features = features_tensor[indices]
        interpolated_features = (neighbor_features * weights[:, :, None]).sum(dim=1)
        
        batch_size = 256
        predictions = []
        confidences = []
        
        for i in range(0, len(interpolated_features), batch_size):
            batch = interpolated_features[i:i+batch_size].to(device)
            
            with torch.no_grad():
                outputs = model.fc(batch) if hasattr(model, 'fc') else batch
                probs = torch.softmax(outputs, dim=1)
                conf, pred = probs.max(dim=1)
                
                predictions.append(pred.cpu())
                confidences.append(conf.cpu())
        
        Z_pred = torch.cat(predictions).numpy().reshape(xx.shape)
        Z_conf = torch.cat(confidences).numpy().reshape(xx.shape)
        
        def plot_fn(ax):
            contour = ax.contourf(xx, yy, Z_conf, levels=20, cmap='RdYlGn', alpha=0.6)
            ax.contour(xx, yy, Z_pred, levels=self.class_num-1, 
                      colors='black', linewidths=0.5, alpha=0.3)
            
            scatter = ax.scatter(
                embedded[:, 0], embedded[:, 1],
                c=targets_tensor.numpy(),
                cmap='tab10' if self.class_num <= 10 else 'hsv',
                s=20,
                edgecolors='black',
                linewidth=0.5,
                alpha=0.8
            )
            
            plt.colorbar(contour, ax=ax, label=self.t['confidence'])
            
            title = self.t['boundary_title']
            if name:
                title += f' - {name}'
            ax.set_title(title)
            
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_fn(ax)
        
        return AnalysisResult(
            fig=fig, ax=ax, 
            data={'predictions': Z_pred, 'confidence': Z_conf, 'embedded': embedded},
            plot_func=plot_fn
        )
