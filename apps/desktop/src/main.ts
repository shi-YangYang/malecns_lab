// The presentation process has no preload, controller or simulator. Keep the
// existing workbench URL unchanged: its IPC trust check requires an exact URL.
const localWorkbenchPreview = location.protocol === 'http:' && location.hostname === '127.0.0.1' && location.hash === '#workbench';
if (location.protocol === 'workbench:' || localWorkbenchPreview) void import('./workbench');
else void import('./brain-presentation');
