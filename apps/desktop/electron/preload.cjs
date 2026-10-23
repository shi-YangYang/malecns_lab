const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('workbench', Object.freeze({
  prepare: () => ipcRenderer.invoke('experiment:prepare'),
  start: () => ipcRenderer.invoke('experiment:start'),
  stop: () => ipcRenderer.invoke('experiment:stop'),
  getSnapshot: () => ipcRenderer.invoke('experiment:snapshot'),
  listRecords: () => ipcRenderer.invoke('records:list'),
  openRecord: id => ipcRenderer.invoke('records:open', id),
  readBatch: (id, sequence) => ipcRenderer.invoke('records:batch', id, sequence),
  setSimulationBounds: bounds => ipcRenderer.invoke('simulation:bounds', bounds),
  setSimulationVisible: visible => ipcRenderer.invoke('simulation:visible', visible),
}));
