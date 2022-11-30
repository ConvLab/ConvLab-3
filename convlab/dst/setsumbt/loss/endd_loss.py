import torch
from torch.nn import Module
from torch.nn.functional import kl_div

EPS = torch.finfo(torch.float32).eps


@torch.no_grad()
def compute_mkl(ensemble_mean_probs: torch.Tensor, ensemble_logprobs: torch.Tensor) -> torch.Tensor:
    """
    Computing MKL in ensemble.

    Args:
        ensemble_mean_probs (Tensor): Marginal predictive distribution of the ensemble
        ensemble_logprobs (Tensor): Log predictive distributions of individual ensemble members

    Returns:
        mkl (Tensor): MKL
    """
    mkl = kl_div(ensemble_logprobs, ensemble_mean_probs.unsqueeze(1).expand_as(ensemble_logprobs),reduction='none')
    return mkl.sum(-1).mean(1)


@torch.no_grad()
def compute_ensemble_stats(ensemble_probs: torch.Tensor) -> dict:
    """
    Compute a range of ensemble uncertainty measures

    Args:
        ensemble_probs (Tensor): Predictive distributions of the ensemble members

    Returns:
        stats (dict): Dictionary of ensemble uncertainty measures
    """
    ensemble_mean_probs = ensemble_probs.mean(dim=1)
    num_classes = ensemble_probs.size(-1)
    ensemble_logprobs = torch.log(ensemble_probs + (1e-4 / num_classes))

    entropy_of_expected = torch.distributions.Categorical(probs=ensemble_mean_probs).entropy()
    expected_entropy = torch.distributions.Categorical(probs=ensemble_probs).entropy().mean(dim=1)
    mutual_info = entropy_of_expected - expected_entropy

    mkl = compute_mkl(ensemble_mean_probs, ensemble_logprobs)

    ensemble_precision = (num_classes - 1) / (2 * mkl.unsqueeze(1) + EPS)

    stats = {
        'probs': ensemble_probs,
        'mean_probs': ensemble_mean_probs,
        'logprobs': ensemble_logprobs,
        'mkl': mkl,
        'precision': ensemble_precision,
        'entropy_of_expected': entropy_of_expected,
        'mutual_info': mutual_info
    }
    return stats


def entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute entropy in a predictive distribution

    Args:
        probs (Tensor): Predictive distributions
        dim (int): Dimension representing the predictive probabilities for a single prediction

    Returns:
        entropy (Tensor): Entropy
    """
    return -(probs * (probs + EPS).log()).sum(dim=dim)


def compute_dirichlet_uncertainties(dirichlet_params: torch.Tensor,
                                    precisions: torch.Tensor,
                                    expected_dirichlet: torch.Tensor) -> tuple:
    """
    Function which computes measures of uncertainty for Dirichlet model.

    Args:
        dirichlet_params (Tensor): Dirichlet concentration parameters.
        precisions (Tensor): Dirichlet Precisions
        expected_dirichlet (Tensor): Probabities of expected categorical under Dirichlet.

    Returns:
        stats (tuple): Token level uncertainties
    """
    batch_size, n_classes = dirichlet_params.size()

    entropy_of_expected = entropy(expected_dirichlet)

    expected_entropy = -expected_dirichlet * (torch.digamma(dirichlet_params + 1) - torch.digamma(precisions + 1))
    expected_entropy = expected_entropy.sum(dim=-1)

    mutual_information = torch.log(expected_dirichlet + EPS) - torch.digamma(dirichlet_params + 1 + EPS)
    mutual_information += torch.digamma(precisions + 1 + EPS)
    mutual_information *= -(expected_dirichlet + EPS)
    mutual_information = mutual_information.sum(dim=-1)

    epkl = (n_classes - 1) / precisions.squeeze(-1)

    mkl = torch.log(expected_dirichlet + EPS) - torch.digamma(dirichlet_params + EPS)
    mkl += torch.digamma(precisions + EPS)
    mkl *= expected_dirichlet
    mkl = mkl.sum(dim=-1)

    stats = (entropy_of_expected.clamp(min=0), expected_entropy.clamp(min=0), mutual_information.clamp(min=0))
    stats += (epkl.clamp(min=0), mkl.clamp(min=0))

    return stats


def get_dirichlet_parameters(logits: torch.Tensor,
                             parametrization,
                             add_to_alphas: float = 0,
                             dtype=torch.double) -> tuple:
    """
    Get dirichlet parameters from model logits

    Args:
        logits (Tensor): Model logits
        parametrization (function): Mapping from logits to concentration parameters
        add_to_alphas (float): Addition constant for stability
        dtype (data type): Data type of the parameters

    Return:
        params (tuple): Concentration and precision parameters of the model Dirichlet
    """
    max_val = torch.finfo(dtype).max / logits.size(-1) - 1
    alphas = torch.clip(parametrization(logits.to(dtype=dtype)) + add_to_alphas, max=max_val)
    precision = torch.sum(alphas, dim=-1, dtype=dtype)
    return alphas, precision


def logits_to_mutual_info(logits: torch.Tensor) -> torch.Tensor:
    """
    Map modfel logits to mutual information of model Dirichlet

    Args:
        logits (Tensor): Model logits

    Returns:
        mutual_information (Tensor): Mutual information of the model Dirichlet
    """
    alphas, precision = get_dirichlet_parameters(logits, torch.exp, 1.0)

    normalized_probs = alphas / precision.unsqueeze(1)

    _, _, mutual_information, _, _ = compute_dirichlet_uncertainties(alphas, precision.unsqueeze(1), normalized_probs)
    
    return mutual_information


class RKLDirichletMediatorLoss(Module):
    """Reverse KL Dirichlet Mediator Loss (https://arxiv.org/abs/2105.06987)"""

    def __init__(self,
                 model_offset: float = 1.0,
                 target_offset: float = 1,
                 ignore_index: int = -1,
                 parameterization=torch.exp):
        """
        Args:
            model_offset (float): Offset of model Dirichlet for stability
            target_offset (float): Offset of target Dirichlet for stability
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            parameterization (function): Mapping from logits to concentration parameters
        """
        super(RKLDirichletMediatorLoss, self).__init__()

        self.model_offset = model_offset
        self.target_offset = target_offset
        self.ignore_index = ignore_index
        self.parameterization = parameterization

    def logits_to_mutual_info(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Map modfel logits to mutual information of model Dirichlet

        Args:
            logits (Tensor): Model logits

        Returns:
            mutual_information (Tensor): Mutual information of the model Dirichlet
        """
        return logits_to_mutual_info(logits)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (Tensor): Model logits
            targets (Tensor): Ensemble predictive distributions

        Returns:
            loss (Tensor): RKL dirichlet mediator loss value
        """

        # Remove padding
        turns = torch.where(targets[:, 0, 0] != self.ignore_index)[0]
        logits = logits[turns]
        targets = targets[turns]

        ensemble_stats = compute_ensemble_stats(targets)

        alphas, precision = get_dirichlet_parameters(logits, self.parameterization, self.model_offset)

        normalized_probs = alphas / precision.unsqueeze(1)

        stats = compute_dirichlet_uncertainties(alphas, precision.unsqueeze(1), normalized_probs)
        entropy_of_expected, expected_entropy, mutual_information, epkl, mkl = stats

        stats = {
            'alpha_min': alphas.min(),
            'alpha_mean': alphas.mean(),
            'precision': precision,
            'entropy_of_expected': entropy_of_expected,
            'mutual_info': mutual_information,
            'mkl': mkl,
        }

        num_classes = alphas.size(-1)

        ensemble_precision = ensemble_stats['precision']

        ensemble_precision += self.target_offset * num_classes
        ensemble_probs = ensemble_stats['mean_probs']

        expected_kl_term = torch.digamma(alphas + EPS) - torch.digamma(precision.unsqueeze(-1) + EPS)
        expected_kl_term = -1.0 * torch.sum(ensemble_probs * expected_kl_term, dim=-1)
        assert torch.isfinite(expected_kl_term).all(), (torch.max(alphas), torch.max(precision), alphas.dtype)

        differential_negentropy_term_ = torch.digamma(alphas + EPS) - torch.digamma(precision.unsqueeze(-1) + EPS)
        differential_negentropy_term_ *= alphas - 1.0
        differential_negentropy_term = torch.sum(torch.lgamma(alphas + EPS), dim=-1) - torch.lgamma(precision + EPS)
        differential_negentropy_term -= torch.sum(differential_negentropy_term_, dim=-1)
        assert torch.isfinite(differential_negentropy_term).all()

        loss = expected_kl_term - differential_negentropy_term / ensemble_precision.squeeze(-1)
        assert torch.isfinite(loss).all()

        return torch.mean(loss), stats, ensemble_stats


class BinaryRKLDirichletMediatorLoss(RKLDirichletMediatorLoss):
    """Reverse KL Dirichlet Mediator Loss (https://arxiv.org/abs/2105.06987)"""

    def __init__(self,
                 model_offset: float = 1.0,
                 target_offset: float = 1,
                 ignore_index: int = -1,
                 parameterization=torch.exp):
        """
        Args:
            model_offset (float): Offset of model Dirichlet for stability
            target_offset (float): Offset of target Dirichlet for stability
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            parameterization (function): Mapping from logits to concentration parameters
        """
        super(BinaryRKLDirichletMediatorLoss, self).__init__(model_offset, target_offset,
                                                             ignore_index, parameterization)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (Tensor): Model logits
            targets (Tensor): Ensemble predictive distributions

        Returns:
            loss (Tensor): RKL dirichlet mediator loss value
        """
        # Convert single target probability p to distribution [1-p, p]
        targets = targets.reshape(-1, targets.size(-1), 1)
        targets = torch.cat([1 - targets, targets], -1)
        targets[targets[:, 0, 1] == self.ignore_index] = self.ignore_index

        # Convert input logits into predictive distribution [1-z, z]
        logits = torch.sigmoid(logits).unsqueeze(1)
        logits = torch.cat((1 - logits, logits), 1)
        logits = -1.0 * torch.log((1 / (logits + 1e-8)) - 1)  # Inverse sigmoid

        return super().forward(logits, targets)
