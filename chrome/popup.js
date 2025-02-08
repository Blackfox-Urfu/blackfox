document.addEventListener('DOMContentLoaded', () => {
  const channelInput = document.getElementById('channelInput');
  const addButton = document.getElementById('addButton');
  const channelList = document.getElementById('channelList');

  if (!channelInput || !addButton || !channelList) {
      console.error("Ошибка: не найдены элементы интерфейса в popup.html.");
      return;
  }

  function loadSettings() {
      chrome.storage.local.get(['excludedChannels'], (data) => {
          const channels = data.excludedChannels || [];
          channelList.innerHTML = '';
          channels.forEach(channel => addChannelToList(channel));
      });
  }

  function saveSettings(channels) {
      chrome.storage.local.set({ excludedChannels: channels }, () => {
          console.log('Настройки сохранены:', channels);
      });
  }

  function addChannelToList(channel) {
      const li = document.createElement('li');
      li.textContent = channel;

      const removeButton = document.createElement('button');
      removeButton.textContent = 'Удалить';
      removeButton.onclick = () => removeChannel(channel);

      li.appendChild(removeButton);
      channelList.appendChild(li);
  }

  function removeChannel(channel) {
      chrome.storage.local.get(['excludedChannels'], (data) => {
          const channels = (data.excludedChannels || []).filter(ch => ch !== channel);
          saveSettings(channels);
          loadSettings();
      });
  }

  addButton.addEventListener('click', () => {
      const newChannel = channelInput.value.trim();
      if (newChannel || newChannel.startsWith('https://web.telegram.org/k/#@')) {
          alert('Введите корректную ссылку на канал имя канала, указанного в шапке. Проще будет через кнопку "Уволить" в самом веб телеграмме.');
          return;
      }

      chrome.storage.local.get(['excludedChannels'], (data) => {
          const channels = data.excludedChannels || [];
          if (!channels.includes(newChannel)) {
              channels.push(newChannel);
              saveSettings(channels);
              loadSettings();
          }
      });
  });

  loadSettings();
});
