import torch
import torch.nn as nn


_MODELS = {}

def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def create_model(config):
  """Create the score model."""
  model_name = config.model.name
  score_model = get_model(model_name)(config)
  score_model = score_model.to(config.device)
  return score_model

@register_model(name='ToyConditionalModel')
class ToyConditionalModel(nn.Module):

    def __init__(self, config):
        super(ToyConditionalModel, self).__init__()
        self.dim = dim = config.data.dim
        hidden_dim = config.model.hidden_dim
        if config.model.activation == 'ReLU':
            self.activation = nn.ReLU()
        elif config.model.activation == 'Softplus':
            self.activation = nn.Softplus()
        else:
            raise Exception("Unknown activation function type")
        self.lin_in = nn.Linear(dim + 1, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.inner_layers1 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(config.model.num_hidden1)])
        self.middle_layer = nn.Linear(hidden_dim + 1, hidden_dim)
        self.inner_layers2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(config.model.num_hidden2)])
        self.lin_out = nn.Linear(hidden_dim + 1, dim)

    def forward(self, x, noise_labels):
        noise_labels = noise_labels[:, None] 
        x = torch.cat((x, noise_labels), dim=1)
        x = self.activation(self.lin_in(x))
        x = self.batch_norm(x)
        for layer in self.inner_layers1:
            x = self.activation(layer(x))
        x = torch.cat((x, noise_labels), dim=1)
        x = self.activation(self.middle_layer(x))
        for layer in self.inner_layers2:
            x = self.activation(layer(x))
        x = torch.cat((x, noise_labels), dim=1)
        x = self.lin_out(x)
        return x
