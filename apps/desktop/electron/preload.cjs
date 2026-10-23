const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('workbench', Object.freeze({
  prepare: () => ipcRenderer.invoke('experiment:prepare'),
  start: () => ipcRenderer.invoke('experiment:start'),
  stop: () => ipcRenderer.invoke('experiment:stop'),
  task: task => ipcRenderer.invoke('experiment:task', task),
  getSnapshot: () => ipcRenderer.invoke('experiment:snapshot'),
  listRecords: () => ipcRenderer.invoke('records:list'),
  openRecord: id => ipcRenderer.invoke('records:open', id),
  readRecordedFrame: (id, sequence) => ipcRenderer.invoke('records:frame', id, sequence),
  readBatch: (id, sequence, cursorIndex) => ipcRenderer.invoke('records:batch', id, sequence, cursorIndex),
  setSimulationBounds: bounds => ipcRenderer.invoke('simulation:bounds', bounds),
  setSimulationVisible: visible => ipcRenderer.invoke('simulation:visible', visible),
}));
