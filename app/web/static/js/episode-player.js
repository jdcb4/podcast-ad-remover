(() => {
    const grid = document.getElementById('episodes-grid');
    if (!grid) return;
    const audio = new Audio();
    audio.preload = 'none';
    let activeId = null;
    let generation = 0;
    let error = '';
    const clock = seconds => {
        const value = Math.floor(Number.isFinite(seconds) ? seconds : 0);
        const hours = Math.floor(value / 3600);
        const minutes = Math.floor(value / 60) % 60;
        return (hours ? `${hours}:${String(minutes).padStart(2, '0')}` : minutes)
            + `:${String(value % 60).padStart(2, '0')}`;
    };
    function render() {
        grid.querySelectorAll('[data-episode-play]').forEach(button => {
            const active = button.dataset.episodePlay === activeId;
            const playing = active && !audio.paused && !audio.ended;
            button.setAttribute('aria-label', playing ? 'Pause episode' : 'Play episode');
            button.setAttribute('aria-pressed', String(playing));
            button.title = playing ? 'Pause' : 'Play';
            button.querySelector('[data-play-icon]').hidden = playing;
            button.querySelector('[data-pause-icon]').hidden = !playing;
            const controls = button.parentElement.querySelector('[data-episode-progress]');
            controls.hidden = !active;
            if (!active) return;
            const seek = controls.querySelector('input');
            const duration = Number.isFinite(audio.duration) ? audio.duration : 0;
            seek.disabled = !duration;
            seek.value = duration ? audio.currentTime / duration * 100 : 0;
            const text = `${clock(audio.currentTime)} / ${clock(duration)}`;
            seek.setAttribute('aria-valuetext', text);
            const time = controls.querySelector('[data-episode-time]');
            if (time.textContent !== text) time.textContent = text;
            const status = controls.querySelector('[data-episode-error]');
            if (status.textContent !== error) status.textContent = error;
        });
    }
    grid.addEventListener('click', async event => {
        const button = event.target.closest('[data-episode-play]');
        if (!button) return;
        if (activeId === button.dataset.episodePlay && !audio.paused) {
            generation++;
            audio.pause();
            render();
            return;
        }
        const attempt = ++generation;
        if (activeId !== button.dataset.episodePlay) {
            audio.pause();
            activeId = button.dataset.episodePlay;
            audio.src = button.dataset.audioUrl;
        }
        error = '';
        if (audio.ended) audio.currentTime = 0;
        render();
        try {
            await audio.play();
        } catch (_) {
            if (attempt !== generation) return;
            error = 'Unable to play audio. Try again or download it.';
        }
        render();
    }, true);
    grid.addEventListener('input', event => {
        if (event.target.dataset.episodeSeek !== activeId) return;
        if (Number.isFinite(audio.duration) && audio.duration > 0) {
            audio.currentTime = Number(event.target.value) / 100 * audio.duration;
            render();
        }
    });
    ['play', 'pause', 'ended', 'timeupdate', 'loadedmetadata', 'durationchange'].forEach(name => {
        audio.addEventListener(name, render);
    });
    audio.addEventListener('error', () => {
        error = 'Unable to play audio. Try again or download it.';
        render();
    });
    // Cards can be replaced by search, filtering or pagination during playback.
    new MutationObserver(render).observe(grid, {childList: true, subtree: true});
    window.addEventListener('pagehide', () => audio.pause());
})();
