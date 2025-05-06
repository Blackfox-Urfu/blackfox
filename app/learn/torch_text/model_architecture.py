import torch.nn as nn

class AdvancedTextClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes=2,
                 dropout=0.3, activation='relu', use_batch_norm=True):
        super(AdvancedTextClassifier, self).__init__()
        layers = []
        prev_size = input_size
        
        # Создание скрытых слоев
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
                
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'elu':
                layers.append(nn.ELU())
                
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
            
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)