import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional
from .base import BaseAnalyzer, AnalysisResult

class GradCAMMap(BaseAnalyzer):
    '''
    Analyzer for generating Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations.
    Inherits from BaseAnalyzer.
    '''

    def __init__(
        self,
        class_info: Union[int, list[str]] = None,
        lang: str = None
    ) -> None:
        '''
        Initializes the GradCAMMap analyzer.

        Args:
            class_info (Union[int, list[str]], optional): Information about classes. Defaults to None.
            lang (str, optional): Language for visualization titles/labels ('en' or 'zh'). Defaults to None.
        '''
        super().__init__(class_info, lang=lang)
    
    @torch.enable_grad()
    def analyse(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        name: str = ''
    ) -> AnalysisResult:
        '''
        Generates the Grad-CAM visualization for a given input image and target class.

        Args:
            model (torch.nn.Module): The classification model.
            target_layer (torch.nn.Module): The specific layer (usually the last convolutional layer) to visualize.
            input_tensor (torch.Tensor): The input image tensor.
            target_class (Optional[int], optional): The target class index. If None, the predicted class is used. Defaults to None.
            name (str, optional): Optional name to append to the plot title. Defaults to ''.

        Returns:
            AnalysisResult: An object containing the generated visualization figure and the raw CAM array.
        '''
        model.eval()
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor.requires_grad = True
        
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())
        
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        output = model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        forward_handle.remove()
        backward_handle.remove()
        
        activation = activations[0]
        gradient = gradients[0]
        
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)
        
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        input_img = input_tensor.squeeze().cpu().detach().numpy()
        if input_img.shape[0] == 3:
            input_img = np.transpose(input_img, (1, 2, 0))
            input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)
        elif input_img.shape[0] == 1:
            input_img = input_img.squeeze()
        
        def plot_fn(ax):
            if len(input_img.shape) == 3:
                ax.imshow(input_img)
            else:
                ax.imshow(input_img, cmap='gray')
            
            cam_resized = np.array(plt.cm.jet(cam)[:, :, :3])
            ax.imshow(cam_resized, alpha=0.5, interpolation='bilinear')
            
            title = self.t['gradcam_title']
            if name:
                title += f' - {name}'
            if self.class_name and target_class < len(self.class_name):
                title += f' ({self.class_name[target_class]})'
            
            ax.set_title(title)
            ax.axis('off')
        
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_fn(ax)
        
        return AnalysisResult(fig=fig, ax=ax, data=cam, plot_func=plot_fn)
