Вот подробный гайд, как собрать и добавить кастомные версии PyTorch и TorchVision в проект на **Poetry** (с виртуальным окружением и зависимостями):

---

## 🔧 **1. Подготовка системы**

Убедись, что у тебя установлены:

* `python3.12` или подходящая версия
* `pip`, `setuptools`, `wheel`
* `Poetry` (`pip install poetry`)
* `git`
* `ninja`, `cmake`, `gcc`, `g++`, `python3.12-dev` и прочее (зависит от дистрибутива)
* CUDA (если хочешь использовать GPU-сборку)

---
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
```

Если ты уже клонировал, то обязательно:

```bash
git submodule sync
git submodule update --init --recursive
```

### ⚙️ Настройка окружения:

если 3.12 нет , то придется установить с помощью pyenv , использовать python вместо python3.12

```bash
python3.12 -m venv venv
source venv/bin/activate

pip install -r requirements.txt  
pip install --upgrade pip setuptools wheel
pip install numpy
```

### 🧱 Сборка и создание `.whl`:

```bash
python setup.py bdist_wheel
```

### 📦 Готовый файл:

Он появится в папке:

```bash
./dist/torch-*.whl
```

---

## 🧩 **3. Сборка torchvision**

### 📥 Клонирование:

```bash
cd ..
git clone https://github.com/pytorch/vision.git
cd vision
```

### ⚙️ Окружение и установка зависимостей:

```bash
python3.12 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install numpy
pip install /path/to/your/torch.whl
```

### 🧱 Сборка `.whl`:

```bash
pip install -e .
```

или если хочешь именно `.whl`:

```bash
python setup.py bdist_wheel
```

📦 Готовый файл появится в:

```bash
./dist/torchvision-*.whl
```

---

## 📁 **4. Добавление кастомных `.whl` в проект Poetry**

### 💼 В проекте `blackfox`:

```bash
cd ~/projects/blackfox
```

### ✍️ Добавление кастомных пакетов:

```bash
poetry add /absolute/path/to/torch-2.x.x.whl
poetry add /absolute/path/to/torchvision-0.x.x.whl
```

> 💡 Лучше указывать **абсолютный путь**, чтобы Poetry не ошибся в URI кодировке (`+` в имени файла и т.п.)

Пример:

```bash
poetry add /home/pesha/projects/pytorch/dist/torch-2.8.0a0+git670dab6-cp312-cp312-linux_x86_64.whl
poetry add /home/pesha/projects/vision/dist/torchvision-0.23.0a0+6473b77-cp312-cp312-linux_x86_64.whl
```

или указывать напрмую в pyproject.toml 

``` toml
[project]
name = "blackfox"
version = "0.1.0"
description = ""
authors = [
    {name = "unser229",email = "nzlobin041@gmail.com"}
]
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "torch @ file:///home/pesha/projects/pytorch/dist/torch-2.8.0a0+git670dab6-cp312-cp312-linux_x86_64.whl",
    "scikit-learn (>=1.7.0,<2.0.0)",
    "joblib (>=1.5.1,<2.0.0)",
    "optuna (>=4.3.0,<5.0.0)",
    "numpy (>=2.3.0,<3.0.0)",
    "seaborn (>=0.13.2,<0.14.0)",
    "nltk (>=3.9.1,<4.0.0)",
    "tqdm (>=4.67.1,<5.0.0)",
    "torchvision @ file:///home/pesha/projects/vision/dist/torchvision-0.23.0a0%2B6473b77-cp312-cp312-linux_x86_64.whl",
    "onnx (>=1.18.0,<2.0.0)",
    "onnxruntime (>=1.22.0,<2.0.0)",
    "fastapi (>=0.115.12,<0.116.0)",
    "uvicorn (>=0.34.3,<0.35.0)",
    "python-multipart (>=0.0.20,<0.0.21)"
]


[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
package-mode = false

```


---

## ✅ **5. Проверка**

После установки:

```bash
poetry run python -c "import torch; print(torch.__version__)"
poetry run python -c "import torchvision; print(torchvision.__version__)"
```

---

## 💡 Дополнительно

* Если хочешь использовать `poetry.lock` как фиксатор версий — просто закоммить его в Git.
* Если проект не должен собираться как библиотека, добавь в `pyproject.toml`:

```toml
[tool.poetry]
package-mode = false
```

---

Если нужно — могу сгенерировать `Makefile` или скрипт, автоматизирующий всё это.
