import torch.nn.functional as F
import math

from codon.base  import *
from dataclasses import dataclass
from codon.exp.ops.manifold import riemannian_manifold_linear, euclidean_manifold_linear


@dataclass
class MainfoldLoss:
    '''
    Dataclass for storing manifold-related loss components.

    Attributes:
        cosine (torch.Tensor): The cosine similarity loss.
        laplacian (torch.Tensor): The Laplacian regularization loss.
    '''
    cosine: torch.Tensor
    laplacian: torch.Tensor

    def factor_loss(self, factor_cos: float = 0.013, factor_lap: float = 0.012) -> torch.Tensor:
        '''
        Calculates the weighted sum of cosine and Laplacian losses.

        Args:
            factor_cos (float): The weight factor for the cosine loss.
            factor_lap (float): The weight factor for the Laplacian loss.

        Returns:
            torch.Tensor: The calculated total loss value.
        '''
        return self.cosine * factor_cos + self.laplacian * factor_lap


class BasicManifoldLinear(BasicModel):
    '''
    Base class for manifold-based neural network layers.

    Attributes:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        k_neighbors (int): Number of nearest neighbors to consider for Laplacian loss.
        weight (nn.Parameter): The learnable weights of the layer.
    '''

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k_neighbors: int = 2,
    ) -> None:
        '''
        Initializes the BasicManifoldLinear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            k_neighbors (int): Number of nearest neighbors for the Laplacian graph.
        '''
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.k_neighbors = min(k_neighbors, out_features - 1)

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
    
    @property
    def loss_cosine(self) -> torch.Tensor:
        '''
        Calculates the cosine similarity penalty loss among the weight vectors.
        
        Returns:
            torch.Tensor: The computed cosine penalty loss.
        '''
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        C = torch.matmul(w_norm, w_norm.T)
        I = torch.eye(self.out_features, device=C.device)

        return torch.sum((C * (1 - I)) ** 2) / (self.out_features * (self.out_features - 1))
    
    @property
    def loss_laplacian(self) -> torch.Tensor:
        '''
        Calculates the Laplacian regularization loss based on k-nearest neighbors.
        
        Returns:
            torch.Tensor: The computed Laplacian regularization loss.
        '''
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        C = torch.matmul(w_norm, w_norm.T)
        I = torch.eye(self.out_features, device=C.device)
        
        _, topk_idx = torch.topk(C, self.k_neighbors + 1, dim=1)
        
        A = torch.zeros_like(C)
        A.scatter_(1, topk_idx, 1.0)
        A = A - I
        A = torch.max(A, A.T)
        
        return torch.sum(A * (1.0 - C)) / torch.sum(A)
    
    def compute_loss(self) -> MainfoldLoss:
        '''
        Computes both the cosine and Laplacian losses and returns them in a MainfoldLoss object.
        
        Returns:
            MainfoldLoss: An object containing the computed cosine and Laplacian losses.
        '''
        w_norm = F.normalize(self.weight, p=2, dim=1)

        C = torch.matmul(w_norm, w_norm.T)
        I = torch.eye(self.out_features, device=C.device)

        loss_cos = torch.sum((C * (1 - I)) ** 2) / (self.out_features * (self.out_features - 1))
        
        _, topk_idx = torch.topk(C, self.k_neighbors + 1, dim=1)
        
        A = torch.zeros_like(C)
        A.scatter_(1, topk_idx, 1.0)
        A = A - I
        A = torch.max(A, A.T)
        
        loss_lap = torch.sum(A * (1.0 - C)) / torch.sum(A)
        
        return MainfoldLoss(cosine=loss_cos, laplacian=loss_lap)


class RiemannianManifoldLinear(BasicManifoldLinear):
    '''
    A linear layer projecting data onto a Riemannian manifold (hypersphere).
    
    Attributes:
        kappa (nn.Parameter): Concentration parameter for the von Mises-Fisher (vMF) distribution.
        lambda_rate (nn.Parameter): Gravitational attraction coefficient.
        scale (nn.Parameter): Vector amplifier for the hyperspherical network.
        bias (nn.Parameter): Manifold bias vector.
    '''

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kappa_init: float = 2.0,
        lambda_init: float = 0.1,
        scale_init: float = 15.0,
        k_neighbors: int = 2,
        rule: str = 'near'
    ) -> None:
        '''
        Initializes the RiemannianManifoldLinear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            kappa_init (float): Initial value for the vMF concentration parameter.
            lambda_init (float): Initial value for the gravitational attraction coefficient.
            scale_init (float): Initial value for the vector amplifier scale.
            k_neighbors (int): Number of nearest neighbors for the Laplacian graph.
            rule (str): Attraction rule, either 'near' or 'far'.
        '''
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            k_neighbors=k_neighbors
        )

        self.kappa_init = kappa_init
        self.lambda_init = lambda_init
        self.scale_init = scale_init
        self.rule = rule.lower()

        if not self.rule in ['far', 'near']:
            raise ValueError(f"Invalid rule: {self.rule}, must be 'far' or 'near'")
        
        # Concentration parameter for the vMF distribution
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))
        
        # Gravitational attraction coefficient
        self.lambda_rate = nn.Parameter(torch.tensor(float(lambda_init)))
        
        # Vector amplifier for the hyperspherical network
        self.scale = nn.Parameter(torch.ones(out_features) * scale_init)

        # Manifold bias vector
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''
        Resets the parameters of the layer.
        '''
        nn.init.normal_(self.weight, 0, 0.01)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Defines the computation performed at every call.

        Args:
            input_tensor (torch.Tensor): The input data with shape (batch_size, in_features).

        Returns:
            torch.Tensor: The output data with shape (batch_size, out_features).
        '''
        x_norm = F.normalize(input_tensor, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # 2. Calculate the cosine similarity matrix c = <x, W>
        # cosine: [batch_size, out_features]
        cosine = F.linear(x_norm, w_norm)
        cosine = torch.clamp(cosine, -1.0 + 1e-6, 1.0 - 1e-6)
        
        # 3. Calculate the geodesic distance (angle theta)
        theta = torch.acos(cosine)
        
        # 4. Compute the vMF gravitational field
        exp = torch.exp(self.kappa * (cosine - 1.0))
        if self.rule == 'far': 
            attraction = 1.0 - exp
        else:
            attraction = exp
        
        # 5. Gravitational pullback on the Riemannian geodesic (analytical solution)
        # effective_theta = theta * (1 - lambda * attraction)
        safe_lambda = torch.clamp(self.lambda_rate, 1e-6, 1.0 - 1e-4)
        effective_theta = theta * (1.0 - safe_lambda * attraction)
        
        # 6. Calculate the final manifold projection
        output = self.scale * torch.cos(effective_theta) + self.bias
        
        return output


class EuclideanManifoldLinear(BasicManifoldLinear):
    '''
    A linear layer simulating a manifold structure in the Euclidean space.
    
    Attributes:
        tau (nn.Parameter): Temperature or radius parameter for the basin of attraction.
        lambda_rate (nn.Parameter): Gravitational strength parameter.
        bias (nn.Parameter): Translation bias vector.
    '''

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_init: float = 5.0,
        lambda_init: float = 0.5,
        k_neighbors: int = 2,
        rule: str = 'near'
    ) -> None:
        '''
        Initializes the EuclideanManifoldLinear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            tau_init (float): Initial value for the basin temperature/radius.
            lambda_init (float): Initial value for the gravitational strength.
            k_neighbors (int): Number of nearest neighbors for the Laplacian graph.
            rule (str): Attraction rule, either 'near' or 'far'.
        '''
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            k_neighbors=k_neighbors
        )
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.rule = rule.lower()
        if not self.rule in ['far', 'near']:
            raise ValueError(f"Invalid rule: {self.rule}, must be 'far' or 'near'")
        
        # Temperature/radius of the basin of attraction
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))

        # Gravitational strength
        self.lambda_rate = nn.Parameter(torch.tensor(float(lambda_init)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''
        Resets the parameters of the layer using Kaiming uniform initialization.
        '''
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Defines the computation performed at every call.

        Args:
            input_tensor (torch.Tensor): The input data with shape (batch_size, in_features).

        Returns:
            torch.Tensor: The output data with shape (batch_size, out_features).
        '''
        # 1. Base linear projection
        # base_proj: [batch_size, out_features]
        base_proj = F.linear(input_tensor, self.weight)
        
        # 2. Ultrafast calculation of squared L2 distance using algebraic expansion
        # ||x - W||^2 = ||x||^2 + ||W||^2 - 2<x, W>
        x_sq = torch.sum(input_tensor ** 2, dim=1, keepdim=True)           # [batch_size, 1]
        w_sq = torch.sum(self.weight ** 2, dim=1).unsqueeze(0)             # [1, out_features]
        
        dist_sq = x_sq + w_sq - 2 * base_proj
        dist_sq = torch.clamp(dist_sq, min=1e-6)
        
        # 3. Compute the attraction index
        exp = torch.exp(-dist_sq / (self.tau ** 2 + 1e-8))
        if self.rule == 'far':
            attraction = 1.0 - exp
        else:
            attraction = exp
        
        # 4. Gravitational correction
        safe_lambda = torch.clamp(self.lambda_rate, 1e-6, 1.0 - 1e-4)
        correction = safe_lambda * attraction * (w_sq - base_proj)
        
        # 5. Combine outputs and add the translation bias
        output = base_proj + correction + self.bias
        
        return output
