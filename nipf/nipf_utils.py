import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def compute_marginals(contingency_table):
  """
  Computes the marginals of a contingency table of arbitrary dimension.
  Returns:
    A list of tensors, where each tensor represents the marginals for a particular dimension.
  """
  marginals = []
  for axis in range(contingency_table.dim()):
    axis_sum = contingency_table.sum(dim=tuple(i for i in range(contingency_table.dim()) if i != axis))
    marginals.append(axis_sum)

  return marginals


class NeuralIPF(nn.Module):
    """
    Neural Iterative Proportional Fitting (NIPF) model.
    This model learns to adjust a seed tensor to match target marginals
    by learning scaling factors for each dimension.
    """
    def __init__(self, seed_tensor):
        super().__init__()
        self.seed = seed_tensor.float()
        self.shape = seed_tensor.shape
        self.log_scales = nn.ParameterList([
            nn.Parameter(torch.zeros(dim_size)) for dim_size in self.shape
        ])

    def forward(self):
        scaled = self.seed
        for axis, log_s in enumerate(self.log_scales):
            scale = torch.exp(log_s)
            shape = [1] * len(self.shape)
            shape[axis] = -1
            scaled = scaled * scale.view(shape)  # broadcasting
        return scaled

def marginal_loss_pdim(pred_tensor, target_marginals):
    loss = 0
    ndim = pred_tensor.dim()
    for axis in range(ndim):
        axes_to_sum = tuple(i for i in range(ndim) if i != axis)
        marginal = pred_tensor.sum(dim=axes_to_sum)
        loss += torch.mean((marginal - target_marginals[axis]) ** 2)
    return loss


# Train the Neural IPF model with given seed tensor and target marginals
def nipf(seed_tensor, target_marginals, lr=0.01, epochs=2000, tol=1e-6):
    """
    Trains the Neural IPF model to adjust the seed tensor to match target marginals.
    Parameters:
        seed_tensor (torch.Tensor): Initial tensor to adjust.
        target_marginals (list of torch.Tensor): List of target marginal tensors for each dimension.
        lr (float): Learning rate for the optimizer.
        epochs (int): Maximum number of training epochs.
        tol (float): Tolerance for early stopping based on loss.
    Returns:
        adjusted_tensor (torch.Tensor): The adjusted tensor after training.
        loss_trace (list): List of loss values recorded during training.
    """
    if not isinstance(seed_tensor, torch.Tensor):
        raise ValueError("seed_tensor must be a torch.Tensor")
    if not isinstance(target_marginals, list) or not all(isinstance(m, torch.Tensor) for m in target_marginals):
        raise ValueError("target_marginals must be a list of torch.Tensor")
    if len(target_marginals) != seed_tensor.dim():
        raise ValueError("Length of target_marginals must match the number of dimensions in seed_tensor")
    
    model = NeuralIPF(seed_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_trace = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        adjusted = model()
        loss = marginal_loss_pdim(adjusted, target_marginals)
        loss.backward()
        optimizer.step()
        loss_trace.append(loss.item())

        if loss.item() < tol:
            break
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1} | Loss: {loss.item():.6e}")

    return model().detach(), loss_trace


# Modified NIPF to return the scaling factors for in in the network inference problem
def nipf2(seed_tensor, target_marginals, lr=0.01, epochs=2000, tol=1e-6):
    """
    Trains the Neural IPF model to adjust the seed tensor to match target marginals
    and returns the learned scaling factors.
    """
    if not isinstance(seed_tensor, torch.Tensor):
        raise ValueError("seed_tensor must be a torch.Tensor")
    if not isinstance(target_marginals, list) or not all(isinstance(m, torch.Tensor) for m in target_marginals):
        raise ValueError("target_marginals must be a list of torch.Tensor")
    if len(target_marginals) != seed_tensor.dim():
        raise ValueError("Length of target_marginals must match the number of dimensions in seed_tensor")
    
    model = NeuralIPF(seed_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_trace = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        adjusted = model()
        loss = marginal_loss_pdim(adjusted, target_marginals)
        loss.backward()
        optimizer.step()
        loss_trace.append(loss.item())

        if loss.item() < tol:
            break
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1} | Loss: {loss.item():.6e}")

    # Recover learned scaling factors
    scale_factors = [torch.exp(log_s.detach()) for log_s in model.log_scales]

    return model().detach(), scale_factors, loss_trace



def plot_loss_trace(loss_trace, save_path=None, file_type='png', dpi=300):
    """
    Plots the training loss trace for Neural IPF, annotated with min loss and convergence epoch.
    """
    plt.figure(figsize=(8, 5))
    epochs = range(len(loss_trace))
    plt.plot(epochs, loss_trace, color='royalblue', linewidth=2, label='Loss')

    # Annotate minimum loss
    min_epoch = int(np.argmin(loss_trace))
    min_loss = loss_trace[min_epoch]
    plt.scatter(min_epoch, min_loss, color='red', zorder=5)
    plt.text(min_epoch, min_loss, f'Min: {min_loss:.2e}\n@ Epoch {min_epoch}', 
             fontsize=10, ha='left', va='bottom', color='red')

    # plt.title("Neural IPF Training Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True)
    plt.legend()

    # Save figure if path is specified
    if save_path:
        full_path = f"{save_path}.{file_type}" if not save_path.endswith(f".{file_type}") else save_path
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {full_path}")

    plt.show()
