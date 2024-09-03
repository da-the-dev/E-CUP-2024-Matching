import torch
import torch.nn.functional as F     

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Compute Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Contrastive loss
        loss = torch.mean(
            (label)
            * torch.pow(euclidean_distance, 2)  # Similar pairs: distance squared
            + (1 - label)
            * torch.pow(
                F.relu(self.margin - euclidean_distance), 2
            )  # Dissimilar pairs: margin - distance squared
        )
        return loss