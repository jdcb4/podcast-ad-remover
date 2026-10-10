(() => {
  const page = document.querySelector('[data-stats-page]');
  if (!page) return;
  let scope = 'mine';
  let period = page.dataset.initialPeriod || 'all';
  function render() {
    for (const [attribute, selected] of [['scope', scope], ['period', period]]) {
      page.querySelectorAll(`[data-stats-${attribute}]`).forEach(button => {
        const active = button.dataset[attribute === 'scope' ? 'statsScope' : 'statsPeriod'] === selected;
        button.setAttribute('aria-pressed', String(active));
        button.classList.toggle('btn-primary', active);
        button.classList.toggle('btn-secondary', !active);
      });
    }
    page.querySelectorAll('[data-stats-collection]').forEach(panel => {
      panel.hidden = panel.dataset.statsCollection !== scope;
    });
    page.querySelectorAll('[data-stats-history]').forEach(panel => {
      panel.hidden = panel.dataset.statsHistory !== period;
    });
  }
  page.addEventListener('click', event => {
    const button = event.target.closest('[data-stats-scope], [data-stats-period]');
    if (!button || !page.contains(button)) return;
    if (button.dataset.statsScope) scope = button.dataset.statsScope;
    if (button.dataset.statsPeriod) period = button.dataset.statsPeriod;
    render();
  });
  render();
})();
