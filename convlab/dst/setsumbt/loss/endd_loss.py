import torch

EPS = torch.finfo(torch.float32).eps

@torch.no_grad()
def compute_mkl(ensemble_probs, ensemble_mean_probs, ensemble_logprobs):
    mkl = torch.nn.functional.kl_div(ensemble_logprobs, ensemble_mean_probs.unsqueeze(1).expand_as(ensemble_probs),
                                     reduction='none').sum(-1).mean(1)
    return mkl

@torch.no_grad()
def compute_ensemble_stats(ensemble_logits):
    # ensemble_probs = torch.softmax(ensemble_logits, dim=-1)
    # ensemble_mean_probs = ensemble_probs.mean(dim=1)
    # ensemble_logprobs = torch.log_softmax(ensemble_logits, dim=-1)
    ensemble_probs = ensemble_logits
    ensemble_mean_probs = ensemble_probs.mean(dim=1)
    num_classes = ensemble_logits.size(-1)
    ensemble_logprobs = torch.log(ensemble_logits + (1e-4 / num_classes))

    entropy_of_expected = torch.distributions.Categorical(probs=ensemble_mean_probs).entropy()
    expected_entropy = torch.distributions.Categorical(probs=ensemble_probs).entropy().mean(dim=1)
    mutual_info = entropy_of_expected - expected_entropy

    mkl = compute_mkl(ensemble_probs, ensemble_mean_probs, ensemble_logprobs)

    # num_classes = ensemble_logits.size(-1)

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

def entropy(probs, dim: int = -1):
    return -(probs * (probs + EPS).log()).sum(dim=dim)


def compute_dirichlet_uncertainties(dirichlet_params, precisions, expected_dirichlet):
    """
    Function which computes measures of uncertainty for Dirichlet model.
    :param dirichlet_params:  Tensor of size [batch_size, n_classes] of Dirichlet concentration parameters.
    :param precisions: Tensor of size [batch_size, 1] of Dirichlet Precisions
    :param expected_dirichlet: Tensor of size [batch_size, n_classes] of probablities of expected categorical under Dirichlet.
    :return: Tensors of token level uncertainties of size [batch_size]
    """
    batch_size, n_classes = dirichlet_params.size()

    entropy_of_expected = entropy(expected_dirichlet)

    expected_entropy = (
            -expected_dirichlet * (torch.digamma(dirichlet_params + 1) - torch.digamma(precisions + 1))).sum(dim=-1)

    mutual_information = -((expected_dirichlet + EPS) * (
            torch.log(expected_dirichlet + EPS) - torch.digamma(dirichlet_params + 1 + EPS) + torch.digamma(
        precisions + 1 + EPS))).sum(dim=-1)
    # assert torch.allclose(mutual_information, entropy_of_expected - expected_entropy, atol=1e-4, rtol=0)

    epkl = (n_classes - 1) / precisions.squeeze(-1)

    mkl = (expected_dirichlet * (
            torch.log(expected_dirichlet + EPS) - torch.digamma(dirichlet_params + EPS) + torch.digamma(
        precisions + EPS))).sum(dim=-1)

    return entropy_of_expected.clamp(min=0), \
           expected_entropy.clamp(min=0), \
           mutual_information.clamp(min=0), \
           epkl.clamp(min=0), \
           mkl.clamp(min=0)

def get_dirichlet_parameters(logits, parametrization, add_to_alphas=0, dtype=torch.double):
    max_val = torch.finfo(dtype).max / logits.size(-1) - 1
    alphas = torch.clip(parametrization(logits.to(dtype=dtype)) + add_to_alphas, max=max_val)
    precision = torch.sum(alphas, dim=-1, dtype=dtype)
    return alphas, precision


def logits_to_mutual_info(logits):
    alphas, precision = get_dirichlet_parameters(logits, torch.exp, 1.0)

    unsqueezed_precision = precision.unsqueeze(1)
    normalized_probs = alphas / unsqueezed_precision

    entropy_of_expected, expected_entropy, mutual_information, epkl, mkl = compute_dirichlet_uncertainties(alphas,
                                                                                                           unsqueezed_precision,
                                                                                                           normalized_probs)
    
    # Max entropy is log(K) for K classes. Hence relative MI is calculated as MI/log(K)
    # mutual_information /= torch.log(torch.tensor(logits.size(-1)))
    
    return mutual_information


def rkl_dirichlet_mediator_loss(logits, ensemble_stats, model_offset, target_offset, parametrization=torch.exp):
    turns = torch.where(ensemble_stats[:, 0, 0] != -1)[0]
    logits = logits[turns]
    ensemble_stats = ensemble_stats[turns]
    
    ensemble_stats = compute_ensemble_stats(ensemble_stats)

    alphas, precision = get_dirichlet_parameters(logits, parametrization, model_offset)

    unsqueezed_precision = precision.unsqueeze(1)
    normalized_probs = alphas / unsqueezed_precision

    entropy_of_expected, expected_entropy, mutual_information, epkl, mkl = compute_dirichlet_uncertainties(alphas,
                                                                                                           unsqueezed_precision,
                                                                                                           normalized_probs)

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

    ensemble_precision += target_offset * num_classes
    ensemble_probs = ensemble_stats['mean_probs']

    expected_KL_term = -1.0 * torch.sum(ensemble_probs * (torch.digamma(alphas + EPS)
                                                          - torch.digamma(precision.unsqueeze(-1) + EPS)), dim=-1)
    assert torch.isfinite(expected_KL_term).all(), (torch.max(alphas), torch.max(precision), alphas.dtype)

    differential_negentropy_term = torch.sum(torch.lgamma(alphas + EPS), dim=-1) - torch.lgamma(precision + EPS) \
                                   - torch.sum(
        (alphas - 1) * (torch.digamma(alphas + EPS) - torch.digamma(precision.unsqueeze(-1) + EPS)), dim=-1)
    assert torch.isfinite(differential_negentropy_term).all()

    cost = expected_KL_term - differential_negentropy_term / ensemble_precision.squeeze(-1)

    assert torch.isfinite(cost).all()
    return torch.mean(cost), stats, ensemble_stats

