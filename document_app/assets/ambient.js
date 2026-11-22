/* Only our own motion state; no widget DOM, timers, requests or reruns. */
(() => {
  const key = '__documentQaAmbient';
  if (window[key]) { window[key].sync(); return; }
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const sync = () => {
    document.documentElement.dataset.docqaMotion =
      document.hidden || reduced.matches ? 'paused' : 'running';
  };
  document.addEventListener('visibilitychange', sync);
  reduced.addEventListener('change', sync);
  window[key] = { sync };
  sync();
})();
