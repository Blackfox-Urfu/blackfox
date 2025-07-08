document.addEventListener('DOMContentLoaded', () => {
    // Основные элементы
    const channelSearch = document.getElementById('channelSearch');
    const addButton = document.getElementById('addButton');
    const channelList = document.getElementById('channelList');
    const exportBtn = document.getElementById('exportBtn');
    const importBtn = document.getElementById('importBtn');
    const importFile = document.getElementById('importFile');
    const downloadLogsBtn = document.getElementById('downloadLogsBtn');

    // Элементы управления
    const enableAdClassification = document.getElementById('enableAdClassification');
    const adSettingsContainer = document.getElementById('adSettingsContainer');
    const displayMode = document.getElementById('displayMode');
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdInput = document.getElementById('thresholdInput');
    const enableNsfwClassification = document.getElementById('enableNsfwClassification');
    const nsfwSettingsContainer = document.getElementById('nsfwSettingsContainer');
    const nsfwAvatarDisplayModeSelect = document.getElementById('nsfwAvatarDisplayMode');

    function toggleSettingsVisibility() {
        if (adSettingsContainer) adSettingsContainer.style.display = enableAdClassification.checked ? 'block' : 'none';
        if (nsfwSettingsContainer) nsfwSettingsContainer.style.display = enableNsfwClassification.checked ? 'block' : 'none';
    }

    function loadSettings() {
        chrome.storage.local.get(null, (data) => {
            updateChannelList(data.excludedChannels || []);
            enableAdClassification.checked = data.enableAdClassification === undefined ? true : data.enableAdClassification;
            displayMode.value = data.displayMode || 'highlight';
            const threshold = data.threshold === undefined ? 0.5 : data.threshold;
            thresholdSlider.value = threshold * 100;
            thresholdInput.value = threshold.toFixed(2);
            enableNsfwClassification.checked = data.enableNsfwClassification === undefined ? true : data.enableNsfwClassification;
            nsfwAvatarDisplayModeSelect.value = data.nsfwAvatarDisplayMode || 'blur';
            toggleSettingsVisibility();
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
        chrome.storage.local.set(settings);
    }

    function removeChannel(channel) {
        chrome.storage.local.get({ excludedChannels: [] }, (data) => {
            const channels = data.excludedChannels.filter(ch => ch !== channel);
            saveSettings({ excludedChannels: channels });
            loadSettings();
        });
    }

    enableAdClassification.addEventListener('change', () => { saveSettings({ enableAdClassification: enableAdClassification.checked }); toggleSettingsVisibility(); });
    enableNsfwClassification.addEventListener('change', () => { saveSettings({ enableNsfwClassification: enableNsfwClassification.checked }); toggleSettingsVisibility(); });
    displayMode.addEventListener('change', () => saveSettings({ displayMode: displayMode.value }));
    nsfwAvatarDisplayModeSelect.addEventListener('change', () => saveSettings({ nsfwAvatarDisplayMode: nsfwAvatarDisplayModeSelect.value }));
    thresholdSlider.addEventListener('input', () => { const value = thresholdSlider.value / 100; thresholdInput.value = value.toFixed(2); saveSettings({ threshold: value }); });
    thresholdInput.addEventListener('input', () => { let value = parseFloat(thresholdInput.value); if (isNaN(value)) value = 0.5; value = Math.max(0, Math.min(1, value)); thresholdSlider.value = value * 100; thresholdInput.value = value.toFixed(2); saveSettings({ threshold: value }); });
    addButton.addEventListener('click', () => { const channelInput = prompt('Введите имя канала (например, "mychannel"):'); if (channelInput) { let channelName = channelInput.trim().replace(/^@/, ''); if (channelName) { chrome.storage.local.get({ excludedChannels: [] }, (data) => { const channels = data.excludedChannels; if (!channels.map(ch => ch.toLowerCase()).includes(channelName.toLowerCase())) { channels.push(channelName); saveSettings({ excludedChannels: channels }); loadSettings(); channelSearch.value = ''; } else { alert(`Канал "${channelName}" уже в списке.`); } }); } } });
    channelSearch.addEventListener('input', (e) => { chrome.storage.local.get({ excludedChannels: [] }, (data) => { updateChannelList(data.excludedChannels, e.target.value); }); });
    exportBtn.addEventListener('click', () => { chrome.storage.local.get({ excludedChannels: [] }, (data) => { const blob = new Blob([JSON.stringify(data.excludedChannels, null, 2)], { type: 'application/json' }); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'bf_excluded_channels.json'; a.click(); URL.revokeObjectURL(url); }); });
    importBtn.addEventListener('click', () => importFile.click());
    importFile.addEventListener('change', (event) => { const file = event.target.files[0]; if (file) { const reader = new FileReader(); reader.onload = (e) => { try { const imported = JSON.parse(e.target.result); if (Array.isArray(imported) && imported.every(ch => typeof ch === 'string')) { if (confirm(`Заменить текущий список на ${imported.length} каналов из файла?`)) { saveSettings({ excludedChannels: imported }); loadSettings(); } } else { alert('Ошибка: файл должен содержать массив строк.'); } } catch (error) { alert('Ошибка чтения файла.'); } }; reader.readAsText(file); } });
    downloadLogsBtn.addEventListener('click', () => { chrome.runtime.sendMessage({ action: "getLogs" }, (response) => { if (response && response.logs) { const logText = response.logs.join('\n'); const blob = new Blob([logText], { type: 'text/plain;charset=utf-8' }); const url = URL.createObjectURL(blob); const a = document.createElement('a'); const timestamp = new Date().toISOString().replace(/[:.]/g, '-'); a.download = `blackfox-extension-logs-${timestamp}.txt`; a.href = url; a.click(); URL.revokeObjectURL(url); } else { alert("Логи пусты или произошла ошибка."); } }); });

    loadSettings();
});