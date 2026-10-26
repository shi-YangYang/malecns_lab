// The presentation process has no preload, controller or simulator. Keep the
// existing workbench URL unchanged: its IPC trust check requires an exact URL.
if (location.protocol === 'workbench:') void import('./workbench');
else void import('./brain-presentation');
