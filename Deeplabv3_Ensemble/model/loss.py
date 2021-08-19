import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing import Optional, List
from catalyst.contrib.nn import \
    DiceLoss, IoULoss, LovaszLossMultiLabel, FocalLossBinary

def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
def soft_jaccard_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return jaccard_score


class JaccardLoss(_Loss):

    def __init__(
        self,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.,
        eps: float = 1e-7,
    ):
        """Implementation of Jaccard loss for image segmentation task.
        It supports binary, multiclass and multilabel cases
        Args:
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(jaccard_coeff)`, otherwise `1 - jaccard_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error 
                (denominator will be always greater or equal to eps)
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super(JaccardLoss, self).__init__()


        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(bs, num_classes, -1)
        #for new iou
        y_true1 = y_true*torch.sum(y_true,dim = 1,keepdim=True)
        # for unbalance data
        y_true1[:,1,:] += y_true[:,1,:]*1
        y_true1[:,3,:] += y_true[:,3,:]*1
        y_true1[:,5,:] += y_true[:,5,:]*1
        y_true1[:,8,:] += y_true[:,8,:]*1
        ##for new iou
        #y_true = y_true*torch.sum(y_true,dim = 1,keepdim=True)
        ####################
        y_pred = y_pred.view(bs, num_classes, -1)
        #y_true1.requires_grad = False
        scores = soft_jaccard_score(y_pred, y_true1.type(y_pred.dtype), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.float()

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()

str2Loss = {
    "bce": nn.BCEWithLogitsLoss,
    "dice": DiceLoss,
    "iou": IoULoss,
    "lovasz": LovaszLossMultiLabel,
    "focal" : FocalLossBinary,
    "JaccardLoss":JaccardLoss
}
class ComposedLossWithLogits(nn.Module):

    def __init__(self, names_and_weights):
        super().__init__()

        assert type(names_and_weights) in (dict, list)

        if isinstance(names_and_weights, dict):
            names_and_weights = names_and_weights.items()

        self.names = []
        self.loss_fns = []
        weights = []

        for name, weight in names_and_weights:
            if weight == 0:
                continue
            self.names.append(name)
            self.loss_fns.append(str2Loss[name]())
            weights.append(weight)

        self.loss_fns = nn.ModuleList(self.loss_fns)

        self.register_buffer('weights', torch.Tensor(weights))
        self.weights /= self.weights.sum()

    def forward(self, logit, target):
        losses = torch.stack([loss_fn(logit, target)
                              for loss_fn in self.loss_fns])

        sumed_loss = (losses * self.weights).sum()

        return sumed_loss, losses
