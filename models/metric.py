import torch

class KLMetric:
    def __init__(self):
        pass

    def compute(self, predictions, targets, matchings):

        pred_latent: Latent = predictions["latent"]
        mu, log_var = pred_latent.mu, pred_latent.log_var
        kl_metric = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())

        return {"kl_metric": kl_metric}