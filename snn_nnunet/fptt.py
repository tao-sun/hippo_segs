"""FPTT running-parameter operations shared by legacy and nnU-Net training."""

import torch


# init before training, lambdas is the gradient \Delta l_t(W_{t+1}, avg_weights is \overline{w}_t.
def init_running_params(model):
        model.avg_weights = {}
        model.lambdas = {}
        for name, param in model.named_parameters():
            model.avg_weights[name] = param.detach().clone().type_as(param)
            model.lambdas[name] = 0.0 * param.detach().clone().type_as(param)


# reset after each epoch
def reset_running_params(model):
    for name, param in model.named_parameters():
        param.data.copy_(model.avg_weights[name].data)


# add a loss item
def regularizer_loss(model, reg_loss, alpha, rho=0.0, _lambda=2.0,):
    # print(f"\nalpha: {model.alpha}, beta: {model.beta}, rho: {rho}, _lambda: {_lambda}")
    for name, param in model.named_parameters():
        reg_loss += (rho-1.) * torch.sum(param * model.lambdas[name])
        reg_loss += _lambda * 0.5 * alpha * torch.sum(torch.square(param - model.avg_weights[name]))
    return reg_loss


# update after each parameter udpate
def update_running_params(model, alpha, beta):
    for name, param in model.named_parameters():
        model.lambdas[name].data.add_(-alpha * (param - model.avg_weights[name]))
        model.avg_weights[name].data.mul_((1.0-beta))
        model.avg_weights[name].data.add_(beta*param-(beta/alpha)*model.lambdas[name])


def export_fptt_tensors(model):
    """Return independent CPU snapshots of the auxiliary FPTT tensors."""
    return {
        state_name: {
            name: tensor.detach().cpu().clone()
            for name, tensor in getattr(model, state_name).items()
        }
        for state_name in ("avg_weights", "lambdas")
    }


def restore_fptt_tensors(model, state):
    """Restore FPTT tensors in each named parameter's device and dtype."""
    parameters = dict(model.named_parameters())
    for state_name in ("avg_weights", "lambdas"):
        saved = state[state_name]
        missing = sorted(set(parameters) - set(saved))
        if missing:
            raise ValueError(
                f"Incomplete FPTT {state_name}: missing {', '.join(missing)}"
            )
        setattr(model, state_name, {
            name: saved[name].to(device=param.device, dtype=param.dtype).clone()
            for name, param in parameters.items()
        })
