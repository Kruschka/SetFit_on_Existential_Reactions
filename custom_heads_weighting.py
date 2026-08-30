import torch
from setfit.modeling import SetFitHead

class PositiveWeightedSetFitHead(SetFitHead):
    """
    A subclass of SetFitHead that incorporates positive class weights into the loss function.
    This is useful for compensating for positive-class imbalance in multilabel classification tasks.
    """

    def __init__(
            self,
            in_features=None,
            out_features=2, 
            temperature=1.0,
            eps=1e-05,
            bias=True,
            device=None,
            multitarget=False,
            positive_weights=None
    ):
        
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            temperature=temperature,
            eps=eps,
            bias=bias,
            device=device,
            multitarget=multitarget
        )

        self.register_buffer("positive_weights", positive_weights)

    def get_loss_fn(self):
        return torch.nn.BCEWithLogitsLoss(
            pos_weight=self.positive_weights
        )

#(in_features: Optional[int] = None, out_features: int = 2, temperature: float = 1.0, eps: float = 1e-05, bias: bool = True, device: Union[torch.device, str, NoneType] = None, multitarget: bool = False) -> None

#(self, in_features: Optional[int] = None, out_features: int = 2, temperature: float = 1.0, eps: float = 1e-05, bias: bool = True, device: Union[torch.device, str, NoneType] = None, multitarget: bool = False) -> None