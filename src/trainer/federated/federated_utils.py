import torch
import logging
import logging
import copy
from typing import List, Dict, Optional, Union, Tuple, Callable

logger = logging.getLogger(__name__)


def generate_permutation_key(
    state_dict: Dict[str, torch.Tensor],
    seed: Optional[int] = None,
    predicate: Optional[Callable[[str, torch.Tensor], bool]] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    """Generate a reproducible random permutation for tensor parameters passing predicate."""
    generator = torch.Generator()
    if seed is None:
        seed = torch.seed()
    generator.manual_seed(seed)

    permutation_key: Dict[str, Optional[torch.Tensor]] = {}
    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor) or param.numel() == 0:
            permutation_key[name] = None
            continue
        if predicate is not None and not predicate(name, param):
            permutation_key[name] = None
            continue
        perm = torch.randperm(param.numel(), generator=generator)
        permutation_key[name] = perm
    return permutation_key


def apply_permutation(state_dict: Dict[str, torch.Tensor], permutation_key: Dict[str, Optional[torch.Tensor]]):
    """Apply a stored permutation key to each tensor parameter."""
    permuted_state: Dict[str, torch.Tensor] = {}
    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            permuted_state[name] = param
            continue
        perm = permutation_key.get(name)
        if perm is None:
            permuted_state[name] = param.clone()
            continue
        flat = param.view(-1)
        device_perm = perm.to(param.device)
        permuted_flat = flat[device_perm]
        permuted_state[name] = permuted_flat.view_as(param)
    return permuted_state


def apply_inverse_permutation(state_dict: Dict[str, torch.Tensor], permutation_key: Dict[str, Optional[torch.Tensor]]):
    """Revert a permutation using its key."""
    restored_state: Dict[str, torch.Tensor] = {}
    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            restored_state[name] = param
            continue
        perm = permutation_key.get(name)
        if perm is None:
            restored_state[name] = param.clone()
            continue
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(perm.numel(), device=perm.device)
        flat = param.view(-1)
        device_inverse = inverse.to(param.device)
        restored_flat = flat[device_inverse]
        restored_state[name] = restored_flat.view_as(param)
    return restored_state


def apply_adversarial_filter(state_dict: Dict[str, torch.Tensor], filter_state: Dict[str, Optional[torch.Tensor]], alpha: float = 1.0):
    """Subtract the revocation filter from each tensor parameter."""
    filtered_state: Dict[str, torch.Tensor] = {}
    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            filtered_state[name] = param
            continue
        filter_param = filter_state.get(name) if filter_state else None
        if filter_param is None:
            filtered_state[name] = param.clone()
            continue
        filtered_state[name] = param - alpha * filter_param.to(param.device)
    return filtered_state


def compute_model_update(initial_state: Dict[str, torch.Tensor], updated_state: Dict[str, torch.Tensor]):
    """Return the difference between updated and initial model parameters."""
    model_update: Dict[str, torch.Tensor] = {}
    for name, updated_tensor in updated_state.items():
        if not isinstance(updated_tensor, torch.Tensor):
            model_update[name] = updated_tensor
            continue
        initial_tensor = initial_state.get(name)
        if initial_tensor is None:
            model_update[name] = updated_tensor.clone()
            continue
        model_update[name] = updated_tensor - initial_tensor.to(updated_tensor.device)
    return model_update


def FedAvg(client_state_dicts, global_model_state_dict=None, client_weights=None):
    num_clients = len(client_state_dicts)
    

    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients
    else:

        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
    
    global_state_dict = {}
    for key in client_state_dicts[0].keys():
        if isinstance(client_state_dicts[0][key], torch.Tensor):
            valid_params = []
            valid_weights = []
            
            for i, state_dict in enumerate(client_state_dicts):
                if state_dict[key].numel() > 0:
                    valid_params.append(state_dict[key])
                    valid_weights.append(client_weights[i])
            
            if valid_params:
                total_valid_weight = sum(valid_weights)
                valid_weights = [w / total_valid_weight for w in valid_weights]
                
                weighted_sum = torch.zeros_like(valid_params[0])
                for param, weight in zip(valid_params, valid_weights):
                    weighted_sum += param * weight
                global_state_dict[key] = weighted_sum
            else:
                if global_model_state_dict is not None:
                    global_state_dict[key] = global_model_state_dict[key]
                    logger.warning(f"All clients have empty parameter for {key}, using global model parameter")
                else:
                    raise ValueError(f"All clients have empty parameter for {key} and no global model provided")
        else:
            global_state_dict[key] = client_state_dicts[0][key]
    
    return global_state_dict


def FedProx(client_state_dicts, global_model_state_dict=None, mu=0.01, client_weights=None):

    return FedAvg(client_state_dicts, global_model_state_dict, client_weights)




def FedAvgM(client_state_dicts: List[Dict], global_model_state_dict: Dict, 
           proxy_dict: Dict = None, round_idx: int = 0, momentum_factor: float = 0.9, 
           tau: float = 1e-3, client_weights: List[float] = None) -> Tuple[Dict, Dict]:
   
    num_clients = len(client_state_dicts)
    
    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients
    else:
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
    
    if proxy_dict is None:
        proxy_dict = {}
        for key in global_model_state_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_model_state_dict[key])
        
    for key in global_model_state_dict.keys():
        delta = torch.zeros_like(global_model_state_dict[key])
        for c in range(num_clients):
            delta += client_weights[c] * (client_state_dicts[c][key] - global_model_state_dict[key])
        
        proxy_dict[key] = momentum_factor * proxy_dict[key] + delta if round_idx > 0 else delta
        global_model_state_dict[key] = global_model_state_dict[key] + proxy_dict[key]
    
    return global_model_state_dict, proxy_dict



def FedAdagrad(client_state_dicts: List[Dict], global_model_state_dict: Dict,
               server_velocity: Dict = None, epsilon: float = 1e-3, tau: float = 1e-3, 
               client_weights: List[float] = None) -> Tuple[Dict, Dict]:
  
    num_clients = len(client_state_dicts)
    
    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients
    else:

        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
    
    if server_velocity is None:
        server_velocity = {}
        for key in global_model_state_dict.keys():
            server_velocity[key] = torch.ones_like(global_model_state_dict[key]) * tau**2

    for key in global_model_state_dict.keys():
        delta = torch.zeros_like(global_model_state_dict[key])
        for c in range(num_clients):
            delta += client_weights[c] * (client_state_dicts[c][key] - global_model_state_dict[key])
        
        server_velocity[key] = server_velocity[key] + torch.square(delta)
        
        global_model_state_dict[key] += epsilon * torch.div(delta, torch.sqrt(server_velocity[key]) + tau)
    
    return global_model_state_dict, server_velocity


def FedYogi(client_state_dicts: List[Dict], global_model_state_dict: Dict,
           proxy_dict: Dict = None, opt_proxy_dict: Dict = None, round_idx: int = 0, 
           beta1: float = 0.9, beta2: float = 0.99, epsilon: float = 1e-3, tau: float = 1e-3, 
           client_weights: List[float] = None) -> Tuple[Dict, Dict, Dict]:

    num_clients = len(client_state_dicts)
    

    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients
    else:

        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
    
    if proxy_dict is None:
        proxy_dict = {}
        for key in global_model_state_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_model_state_dict[key])
    
    if opt_proxy_dict is None:
        opt_proxy_dict = {}
        for key in global_model_state_dict.keys():
            opt_proxy_dict[key] = torch.ones_like(global_model_state_dict[key]) * tau**2

    for key in global_model_state_dict.keys():
        delta = torch.zeros_like(global_model_state_dict[key])
        for c in range(num_clients):
            delta += client_weights[c] * (client_state_dicts[c][key] - global_model_state_dict[key])
        
        proxy_dict[key] = beta1 * proxy_dict[key] + (1 - beta1) * delta if round_idx > 0 else delta
        delta_square = torch.square(proxy_dict[key])
        opt_proxy_dict[key] = opt_proxy_dict[key] - (1-beta2)*delta_square*torch.sign(opt_proxy_dict[key] - delta_square)
        global_model_state_dict[key] += epsilon * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+tau)
    
    return global_model_state_dict, proxy_dict, opt_proxy_dict


def FedAdam(client_state_dicts: List[Dict], global_model_state_dict: Dict, 
           proxy_dict: Dict = None, opt_proxy_dict: Dict = None, round_idx: int = 0, 
           beta1: float = 0.9, beta2: float = 0.99, epsilon: float = 1e-3, tau: float = 1e-3, 
           client_weights: List[float] = None) -> Tuple[Dict, Dict, Dict]:
   
    num_clients = len(client_state_dicts)
    

    if client_weights is None:
        client_weights = [1.0 / num_clients] * num_clients
    else:
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
    

    if proxy_dict is None:
        proxy_dict = {}
        for key in global_model_state_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_model_state_dict[key])
    
    if opt_proxy_dict is None:
        opt_proxy_dict = {}
        for key in global_model_state_dict.keys():
            opt_proxy_dict[key] = torch.ones_like(global_model_state_dict[key]) * tau**2
    
    for key in global_model_state_dict.keys():

        delta = torch.zeros_like(global_model_state_dict[key])
        for c in range(num_clients):
            delta += client_weights[c] * (client_state_dicts[c][key] - global_model_state_dict[key])
        

        proxy_dict[key] = beta1 * proxy_dict[key] + (1 - beta1) * delta if round_idx > 0 else delta
        opt_proxy_dict[key] = beta2 * opt_proxy_dict[key] + (1 - beta2) * torch.square(proxy_dict[key])
        global_model_state_dict[key] += epsilon * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key]) + tau)
    
    return global_model_state_dict, proxy_dict, opt_proxy_dict
