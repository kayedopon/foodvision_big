
import torch
import torchvision

from torch import nn

def create_convnext_tiny_model(num_classes:int,
                               seed:int=42):
  """Creates a ConvNeXt Tiny feature extractor model and transforms

  Args:
      num_classes (int): number of target classes.
      seed (int, optional): random seed value for the output layer.

  Returns:
      model (torch.nn.Module): ConvNeXt Tiny feature extractor model.
      transforms (torchvision.transforms): ConvNeXt Tiny image transforms
  """
  convnext_weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
  transforms = convnext_weights.transforms()
  convnext = torchvision.models.convnext_tiny(weights=convnext_weights)

  for param in convnext.parameters():
    param.requires_grad = False

  convnext.classifier = nn.Sequential(
    
    nn.Flatten(start_dim=1, end_dim=-1),
    nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
    nn.Linear(in_features=768, out_features=num_classes)
  )

  return convnext, transforms
