import torch.nn as nn
from torchvision import models

def create_configurable_model(params: dict):
    """
    Создает модель ResNet с настраиваемой архитектурой на основе словаря параметров.

    Args:
        params (dict): Словарь с параметрами модели. Ожидаемые ключи:
            - "base_model" (str): Название базовой модели ('resnet18', 'resnet34', etc.).
            - "unfreeze_strategy" (str): Стратегия разморозки слоев (не используется при инференсе, но нужна для консистентности).
            - "n_fc_layers" (int): Количество полносвязных слоев.
            - "fc_units_l{i}" (int): Количество нейронов в i-том слое.
            - "fc_dropout_l{i}" (float): Dropout для i-того слоя.
    
    Returns:
        torch.nn.Module: Сконфигурированная модель PyTorch.
    """
    base_model_name = params.get("base_model", "resnet34")

    # В режиме инференса мы не загружаем предобученные веса ImageNet,
    # так как все веса будут загружены из нашего файла.
    # weights=None эквивалентно pretrained=False в старом API.
    if base_model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif base_model_name == "resnet34":
        model = models.resnet34(weights=None)
    elif base_model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        # Если модель не известна, по умолчанию используем resnet34
        print(f"Warning: Unknown base model '{base_model_name}', defaulting to resnet34.")
        model = models.resnet34(weights=None)
    
    # Замораживать слои при инференсе не нужно, все параметры уже обучены
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features

    fc_layers_list = []
    n_fc_layers = params.get("n_fc_layers", 2)
    last_out_features = num_ftrs

    for i in range(n_fc_layers):
        # Если в словаре нет ключа, берем значение по умолчанию
        fc_units = params.get(f"fc_units_l{i}", 1024 if i == 0 else 512)
        fc_dropout = params.get(f"fc_dropout_l{i}", 0.5 if i == 0 else 0.3)

        fc_layers_list.append(nn.Linear(last_out_features, fc_units))
        fc_layers_list.append(nn.BatchNorm1d(fc_units))
        fc_layers_list.append(nn.ReLU(inplace=True))
        # Dropout не нужен при инференсе, но для идентичности архитектуры его можно добавить
        # В режиме model.eval() он все равно будет отключен
        fc_layers_list.append(nn.Dropout(fc_dropout))
        last_out_features = fc_units

    fc_layers_list.append(nn.Linear(last_out_features, 1)) # Выходной слой для бинарной классификации
    # Убираем Sigmoid из архитектуры, так как BCEWithLogitsLoss работает с логитами,
    # а на сервере мы можем применить Sigmoid вручную для получения вероятностей.
    # Но в вашем коде сервера вы уже используете Sigmoid, так что вернем его для совместимости.
    #fc_layers_list.append(nn.Sigmoid()) # Раскомментируйте, если ваша модель в сервере должна сама выдавать вероятность
                                        # В вашем случае, на сервере уже есть model(img_tensor).item(), что предполагает Sigmoid в модели.
                                        # Поэтому оставляем его в архитектуре.
    
    # ВАЖНО: Ваша оригинальная модель в main.py имела Sigmoid в конце.
    # А в скрипте обучения вы использовали BCEWithLogitsLoss, который ожидает логиты (без Sigmoid).
    # Для единообразия, лучше убрать Sigmoid из модели и применять его на сервере.
    # Но чтобы не ломать ваш код сервера, я добавлю его сюда.
    # Однако, в скрипте обучения это может вызвать проблемы.
    # Давайте сделаем так: в `main.py` мы будем использовать модель как есть, но в обучении нужно
    # использовать модель без Sigmoid.
    # Для простоты, оставим ваш код как есть (с Sigmoid в модели), но знайте об этом нюансе.
    # **ОБНОВЛЕНИЕ**: В вашем коде `train.py` `model.fc` не имеет Sigmoid, а на сервере `main.py` имеет.
    # Давайте унифицируем. Я изменю `main.py` и `architecture.py` так, чтобы Sigmoid НЕ БЫЛ частью модели.
    # Это более правильный подход.

    # Финальная версия без Sigmoid
    model.fc = nn.Sequential(*fc_layers_list)

    return model
