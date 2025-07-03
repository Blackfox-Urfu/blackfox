# torch_models.py
import torch
import torch.nn as nn
import torchvision.models as models

class SubModel(nn.Module):
    """Базовый MLP для текстовых и числовых признаков."""
    def __init__(self, input_size, output_size=128, hidden_layers=[256], dropout=0.4):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ImageSubModel(nn.Module):
    """Модель для извлечения признаков из изображений с помощью предобученного ResNet."""
    def __init__(self, output_size=256, train_backbone=False):
        super().__init__()
        # Загружаем предобученную модель resnet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Замораживаем веса, если не хотим их обучать
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Заменяем последний слой (классификатор) на наш собственный
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, output_size)

    def forward(self, x):
        return self.backbone(x)

class MetaLearner(nn.Module):
    """
    Главная модель, которая объединяет выходы от всех "экспертов" (SubModels).
    """
    def __init__(self, text_input_size, features_input_size, 
                 text_emb_size=128, features_emb_size=64, image_emb_size=256,
                 meta_hidden_layers=[256, 128], num_classes=2, dropout=0.5):
        super().__init__()

        # 1. Создаем "экспертов"
        self.text_model = SubModel(text_input_size, output_size=text_emb_size, hidden_layers=[512, 256])
        self.features_model = SubModel(features_input_size, output_size=features_emb_size, hidden_layers=[128])
        self.image_model = ImageSubModel(output_size=image_emb_size, train_backbone=False) # Не обучаем resnet для скорости

        # 2. Создаем обучаемые "токены отсутствия"
        self.text_absence_token = nn.Parameter(torch.randn(1, text_emb_size))
        self.features_absence_token = nn.Parameter(torch.randn(1, features_emb_size))
        self.image_absence_token = nn.Parameter(torch.randn(1, image_emb_size))
        # Можно добавить токены для видео, аудио и т.д. по аналогии

        # 3. Создаем "голову" мета-модели
        total_input_size = text_emb_size + features_emb_size + image_emb_size
        
        meta_layers = []
        prev_size = total_input_size
        for hidden_size in meta_hidden_layers:
            meta_layers.append(nn.Linear(prev_size, hidden_size))
            meta_layers.append(nn.BatchNorm1d(hidden_size))
            meta_layers.append(nn.ReLU())
            meta_layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        meta_layers.append(nn.Linear(prev_size, num_classes))
        self.meta_head = nn.Sequential(*meta_layers)

    def forward(self, batch):
        batch_size = batch['labels'].shape[0]

        # --- Обработка Текста ---
        if 'text' in batch and batch['text'] is not None:
            text_embeddings = self.text_model(batch['text'])
        else:
            text_embeddings = self.text_absence_token.repeat(batch_size, 1)

        # --- Обработка Числовых Признаков ---
        if 'features' in batch and batch['features'] is not None:
            features_embeddings = self.features_model(batch['features'])
        else:
            features_embeddings = self.features_absence_token.repeat(batch_size, 1)

        # --- Обработка Изображений ---
        # Инициализируем выходной тензор токенами отсутствия
        image_embeddings = self.image_absence_token.repeat(batch_size, 1)
        if 'images' in batch and batch['images'] is not None:
            # Получаем эмбеддинги только для тех постов, где есть картинки
            processed_images = self.image_model(batch['images'])
            # Расставляем эмбеддинги по правильным местам в батче
            image_embeddings.index_copy_(0, batch['image_indices'], processed_images)

        # Объединяем все эмбеддинги
        combined_embeddings = torch.cat([text_embeddings, features_embeddings, image_embeddings], dim=1)
        
        # Прогоняем через голову мета-модели для финального предсказания
        logits = self.meta_head(combined_embeddings)
        
        return logits