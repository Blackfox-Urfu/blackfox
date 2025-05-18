document.addEventListener('DOMContentLoaded', () => {
  const channelSearch = document.getElementById('channelSearch');
  const addButton = document.getElementById('addButton');
  const channelList = document.getElementById('channelList');
  const displayMode = document.getElementById('displayMode');
  const thresholdSlider = document.getElementById('thresholdSlider');
  const thresholdInput = document.getElementById('thresholdInput');
  const exportBtn = document.getElementById('exportBtn');
  const importBtn = document.getElementById('importBtn');
  const importFile = document.getElementById('importFile');

  // --- НОВЫЕ ЭЛЕМЕНТЫ ДЛЯ НАСТРОЕК АВАТАРОК ---
  const classifyAvatarsEnabledCheckbox = document.getElementById('classifyAvatarsEnabled');
  const nsfwAvatarDisplayModeSelect = document.getElementById('nsfwAvatarDisplayMode');
  const nsfwModeSettingItem = document.getElementById('nsfwModeSettingItem'); // Контейнер для выбора режима

  // Проверка наличия основных элементов
  if (!channelSearch || !addButton || !channelList || !displayMode || !thresholdSlider || !thresholdInput || !exportBtn || !importBtn || !importFile) {
      console.error("POPUP: Ошибка: не найдены основные элементы интерфейса в popup.html.");
      // return; // Можно раскомментировать, если критично
  }
  // Проверка наличия новых элементов
  if (!classifyAvatarsEnabledCheckbox || !nsfwAvatarDisplayModeSelect || !nsfwModeSettingItem) {
      console.warn("POPUP: Не найдены элементы для управления классификацией аватарок. Функционал будет ограничен.");
  }

  // Функция для управления видимостью настроек режима NSFW
  function toggleNsfwModeVisibility() {
    if (nsfwModeSettingItem && classifyAvatarsEnabledCheckbox) {
        nsfwModeSettingItem.style.display = classifyAvatarsEnabledCheckbox.checked ? 'block' : 'none';
    }
  }

  function loadSettings() {
      chrome.storage.local.get([
          'excludedChannels', 
          'displayMode', 
          'threshold',
          'classifyAvatarsEnabled', // Новый ключ
          'nsfwAvatarDisplayMode'   // Новый ключ
        ], (data) => {
          const channels = data.excludedChannels || [];
          updateChannelList(channels);

          if (displayMode && data.displayMode) {
              displayMode.value = data.displayMode;
          }

          const threshold = data.threshold === undefined ? 0.5 : data.threshold; // Учтем undefined
          if (thresholdSlider) thresholdSlider.value = threshold * 100;
          if (thresholdInput) thresholdInput.value = threshold.toFixed(2);

          // --- ЗАГРУЗКА НАСТРОЕК КЛАССИФИКАЦИИ АВАТАРОК ---
          if (classifyAvatarsEnabledCheckbox) {
            // Значение по умолчанию для classifyAvatarsEnabled - true (как в content.js)
            classifyAvatarsEnabledCheckbox.checked = data.classifyAvatarsEnabled === undefined ? true : data.classifyAvatarsEnabled;
          }
          
          if (nsfwAvatarDisplayModeSelect) {
            // Значение по умолчанию для nsfwAvatarDisplayMode - 'blur' (как в content.js при первой загрузке без сохраненных настроек)
            nsfwAvatarDisplayModeSelect.value = data.nsfwAvatarDisplayMode || 'blur'; 
          }
          toggleNsfwModeVisibility(); // Обновляем видимость при загрузке
      });
  }

  function updateChannelList(channels, searchTerm = '') {
      if (!channelList) return;
      channelList.innerHTML = '';
      channels
          .filter(channel => channel.toLowerCase().includes(searchTerm.toLowerCase()))
          .forEach(channel => {
              const li = document.createElement('li');
              li.textContent = channel;

              const removeButton = document.createElement('button');
              removeButton.textContent = 'Удалить';
              removeButton.className = 'remove-btn';
              removeButton.onclick = () => removeChannel(channel);

              li.appendChild(removeButton);
              channelList.appendChild(li);
          });
  }

  function saveSettings(settings) {
      chrome.storage.local.set(settings, () => {
          // console.log('POPUP: Настройки сохранены:', settings);
      });
  }

  function removeChannel(channel) {
      chrome.storage.local.get(['excludedChannels'], (data) => {
          const channels = (data.excludedChannels || []).filter(ch => ch !== channel);
          saveSettings({ excludedChannels: channels });
          loadSettings(); // Перезагружаем для обновления списка
      });
  }

  if (displayMode) {
    displayMode.addEventListener('change', () => {
        saveSettings({ displayMode: displayMode.value });
    });
  }

  if (thresholdSlider) {
    thresholdSlider.addEventListener('input', () => {
        const value = thresholdSlider.value / 100;
        if (thresholdInput) thresholdInput.value = value.toFixed(2);
        saveSettings({ threshold: value });
    });
  }

  if (thresholdInput) {
    thresholdInput.addEventListener('input', () => {
        let value = parseFloat(thresholdInput.value);
        if (isNaN(value)) value = 0.5; // значение по умолчанию, если ввод некорректен
        if (value < 0) value = 0;
        if (value > 1) value = 1;
        if (thresholdSlider) thresholdSlider.value = value * 100;
        thresholdInput.value = value.toFixed(2); // Обновляем для корректного отображения
        saveSettings({ threshold: value });
    });
  }
  
  if (addButton) {
    addButton.addEventListener('click', () => {
        const channelInput = prompt('Введите имя канала (например, "mychannel") или полную ссылку (например, "https://web.telegram.org/k/#@mychannel"):');
        if (channelInput) {
            let channelName = channelInput.trim();
            // Пытаемся извлечь имя канала из URL
            const urlMatch = channelInput.match(/#@([\w-]+)/);
            if (urlMatch && urlMatch[1]) {
                channelName = urlMatch[1];
            }
            // Убираем возможное начальное @
            if (channelName.startsWith('@')) {
                channelName = channelName.substring(1);
            }

            if (channelName) { // Проверяем, что имя не пустое после обработки
                chrome.storage.local.get(['excludedChannels'], (data) => {
                    const channels = data.excludedChannels || [];
                    const lowerCaseChannelName = channelName.toLowerCase();
                    if (!channels.map(ch => ch.toLowerCase()).includes(lowerCaseChannelName)) {
                        channels.push(channelName); // Сохраняем оригинальное имя, но поиск/сравнение по lowercase
                        saveSettings({ excludedChannels: channels });
                        loadSettings(); // Обновляем список
                        if (channelSearch) channelSearch.value = ''; // Очищаем поиск
                    } else {
                        alert(`Канал "${channelName}" уже в списке исключений.`);
                    }
                });
            } else {
                 alert('Не удалось извлечь имя канала. Введите корректное имя или ссылку.');
            }
        }
    });
  }

  if (channelSearch) {
    channelSearch.addEventListener('input', (e) => {
        chrome.storage.local.get(['excludedChannels'], (data) => {
            const channels = data.excludedChannels || [];
            updateChannelList(channels, e.target.value);
        });
    });
  }

  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
        chrome.storage.local.get(['excludedChannels'], (data) => {
            const channels = data.excludedChannels || [];
            const blob = new Blob([JSON.stringify(channels, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = 'telegram_ad_blocker_excluded_channels.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
    });
  }

  if (importBtn && importFile) {
    importBtn.addEventListener('click', () => {
        importFile.click();
    });

    importFile.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const importedChannels = JSON.parse(e.target.result);
                    if (Array.isArray(importedChannels) && importedChannels.every(ch => typeof ch === 'string')) {
                        if (confirm(`Импортировать ${importedChannels.length} каналов? Текущий список исключений будет заменен.`)) {
                            saveSettings({ excludedChannels: importedChannels });
                            loadSettings();
                            alert('Список каналов успешно импортирован!');
                        }
                    } else {
                        alert('Неверный формат файла! Ожидается JSON-массив строк.');
                    }
                } catch (error) {
                    alert('Ошибка при чтении или обработке файла!');
                    console.error('POPUP: Ошибка импорта:', error);
                }
                importFile.value = ''; // Сброс input file для повторного выбора того же файла
            };
            reader.readAsText(file);
        }
    });
  }

  // --- ОБРАБОТЧИКИ ДЛЯ НОВЫХ ЭЛЕМЕНТОВ УПРАВЛЕНИЯ АВАТАРКАМИ ---
  if (classifyAvatarsEnabledCheckbox) {
    classifyAvatarsEnabledCheckbox.addEventListener('change', () => {
        saveSettings({ classifyAvatarsEnabled: classifyAvatarsEnabledCheckbox.checked });
        toggleNsfwModeVisibility(); // Обновляем видимость выбора режима
    });
  }

  if (nsfwAvatarDisplayModeSelect) {
    nsfwAvatarDisplayModeSelect.addEventListener('change', () => {
        saveSettings({ nsfwAvatarDisplayMode: nsfwAvatarDisplayModeSelect.value });
    });
  }
  // --- КОНЕЦ ОБРАБОТЧИКОВ ДЛЯ АВАТАРОК ---

  loadSettings(); // Первичная загрузка всех настроек
});