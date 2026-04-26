import os
import locale
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union, Optional, Any, Callable

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SYS_LANG = 'zh' if 'zh' in (locale.getdefaultlocale()[0] or '').lower() else 'en'

I18N_DICT = {
    'en': {
        'actual': 'Actual',
        'predicted': 'Predicted',
        'confusion_title': 'Confusion Matrix',
        'similarity_title': 'Cosine Similarity Matrix',
        'tsne_title': 't-SNE Feature',
        'rsa_title': 'Representational Similarity Matrix (RSA)',
        'gradcam_title': 'Grad-CAM Visualization',
        'activation_title': 'Activation Distribution',
        'activation_value': 'Activation Value',
        'mean': 'Mean',
        'sparsity': 'Sparsity',
        'layer_rsa_title': 'Layer-wise RSA',
        'layer': 'Layer',
        'boundary_title': 'Decision Boundary & Confidence',
        'confidence': 'Confidence',
        'cka_title': 'CKA Similarity Matrix',
        'model_a': 'Model A',
        'model_b': 'Model B',
        'selectivity_title': 'Neuron Selectivity (Top-K)',
        'neuron': 'Neuron',
        'selectivity': 'Selectivity',
        'class': 'Class',
        'mean_activation': 'Mean Activation'
    },
    'zh': {
        'actual': '真实类别',
        'predicted': '预测类别',
        'confusion_title': '混淆矩阵',
        'similarity_title': '余弦相似度矩阵',
        'tsne_title': 't-SNE 特征',
        'rsa_title': '表征相似度矩阵 (RSA)',
        'gradcam_title': 'Grad-CAM 可视化',
        'activation_title': '激活值分布',
        'activation_value': '激活值',
        'mean': '均值',
        'sparsity': '稀疏度',
        'layer_rsa_title': '跨层表征相似度',
        'layer': '层',
        'boundary_title': '决策边界与置信度',
        'confidence': '置信度',
        'cka_title': 'CKA 相似度矩阵',
        'model_a': '模型 A',
        'model_b': '模型 B',
        'selectivity_title': '神经元选择性 (Top-K)',
        'neuron': '神经元',
        'selectivity': '选择性',
        'class': '类别',
        'mean_activation': '平均激活值'
    }
}

@dataclass
class AnalysisResult:
    fig: plt.Figure
    ax: plt.Axes
    data: Any = None
    plot_func: Optional[Callable] = None

    def show(self) -> 'AnalysisResult':
        plt.figure(self.fig.number)
        plt.show()
        return self
    
    def save(self, path: str, name: str, fmt: str = 'pdf'):
        if not os.path.isdir(path): os.makedirs(path)
        save_path = os.path.join(path, f'{name}.{fmt}')
        self.fig.savefig(save_path, format=fmt, bbox_inches='tight')
        return save_path

    @staticmethod
    def merge(results: list['AnalysisResult'], title: str = None, path: str = None, show: bool = False) -> 'AnalysisResult':
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
        if n == 1: axes = [axes]
        
        for ax, res in zip(axes, results):
            if res.plot_func is not None:
                res.plot_func(ax)
        
        if title: fig.suptitle(title, fontsize=14)

        plt.tight_layout()
        
        merged_res = AnalysisResult(fig=fig, ax=axes)
        if path:
            merged_res.save(path, 'merged_result')
        if show:
            merged_res.show()
        else:
            plt.close(fig)
            
        return merged_res

class BaseAnalyzer:
    def __init__(self, class_info: Union[int, list[str]] = None, lang: str = None):
        _lang = lang or SYS_LANG
        self.lang = _lang if _lang in I18N_DICT else 'en'
        self.t = I18N_DICT[self.lang]
        
        self.class_num = 0
        self.class_name = []
        
        if class_info is not None:
            if isinstance(class_info, int):
                self.class_num = class_info
                self.class_name = [str(i) for i in range(class_info)]
            else:
                self.class_num = len(class_info)
                self.class_name = class_info

    @property
    def show_annot(self) -> bool:
        return self.class_num <= 20
        
    def get_ticklabels(self) -> list:
        if self.class_num <= 30:
            return self.class_name
        
        step = max(1, self.class_num // 20)
        return [name if i % step == 0 else "" for i, name in enumerate(self.class_name)]

    def _get_figsize(self) -> tuple[float, float]:
        size = max(8.0, min(20.0, self.class_num * 0.4))
        return (size, size * 0.75)