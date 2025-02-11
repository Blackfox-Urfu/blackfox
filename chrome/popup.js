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

  if (!channelSearch || !addButton || !channelList) {
      console.error("Ошибка: не найдены элементы интерфейса в popup.html.");
      return;
  }

  function loadSettings() {
      chrome.storage.local.get(['excludedChannels', 'displayMode', 'threshold'], (data) => {
          const channels = data.excludedChannels || [];
          updateChannelList(channels);

          if (data.displayMode) {
              displayMode.value = data.displayMode;
          }

          const threshold = data.threshold || 0.5;
          thresholdSlider.value = threshold * 100;
          thresholdInput.value = threshold.toFixed(2);
      });
  }

  function updateChannelList(channels, searchTerm = '') {
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
          console.log('Настройки сохранены:', settings);
      });
  }

  function removeChannel(channel) {
      chrome.storage.local.get(['excludedChannels'], (data) => {
          const channels = (data.excludedChannels || []).filter(ch => ch !== channel);
          saveSettings({ excludedChannels: channels });
          loadSettings();
      });
  }

  displayMode.addEventListener('change', () => {
      saveSettings({ displayMode: displayMode.value });
  });

  thresholdSlider.addEventListener('input', () => {
      const value = thresholdSlider.value / 100;
      thresholdInput.value = value.toFixed(2);
      saveSettings({ threshold: value });
  });

  thresholdInput.addEventListener('input', () => {
      let value = parseFloat(thresholdInput.value);
      if (value < 0) value = 0;
      if (value > 1) value = 1;
      thresholdSlider.value = value * 100;
      saveSettings({ threshold: value });
  });

  addButton.addEventListener('click', () => {
      const channelUrl = prompt('Введите ссылку на канал:');
      if (channelUrl && channelUrl.startsWith('https://web.telegram.org/k/#@')) {
          chrome.storage.local.get(['excludedChannels'], (data) => {
              const channels = data.excludedChannels || [];
              if (!channels.includes(channelUrl)) {
                  channels.push(channelUrl);
                  saveSettings({ excludedChannels: channels });
                  loadSettings();
              }
          });
      } else if (channelUrl) {
          alert('Введите корректную ссылку на канал');
      }
  });

  channelSearch.addEventListener('input', (e) => {
      chrome.storage.local.get(['excludedChannels'], (data) => {
          const channels = data.excludedChannels || [];
          updateChannelList(channels, e.target.value);
      });
  });

  // Функция экспорта списка каналов
  exportBtn.addEventListener('click', () => {
      chrome.storage.local.get(['excludedChannels'], (data) => {
          const channels = data.excludedChannels || [];
          const blob = new Blob([JSON.stringify(channels, null, 2)], { type: 'application/json' });
          const url = URL.createObjectURL(blob);
          
          const a = document.createElement('a');
          a.href = url;
          a.download = 'telegram_ad_blocker_channels.json';
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
      });
  });

  // Обработчик кнопки импорта
  importBtn.addEventListener('click', () => {
      importFile.click();
  });

  // Функция импорта списка каналов
  importFile.addEventListener('change', (event) => {
      const file = event.target.files[0];
      if (file) {
          const reader = new FileReader();
          reader.onload = (e) => {
              try {
                  const channels = JSON.parse(e.target.result);
                  if (Array.isArray(channels)) {
                      // Подтверждение импорта
                      if (confirm(`Импортировать ${channels.length} каналов? Текущий список будет заменен.`)) {
                          chrome.storage.local.set({ excludedChannels: channels }, () => {
                              loadSettings();
                              alert('Список каналов успешно импортирован!');
                          });
                      }
                  } else {
                      alert('Неверный формат файла!');
                  }
              } catch (error) {
                  alert('Ошибка при чтении файла!');
                  console.error('Ошибка импорта:', error);
              }
          };
          reader.readAsText(file);
      }
  });

  loadSettings();
});
