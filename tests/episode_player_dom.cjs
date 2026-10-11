const fs = require('fs');
const assert = require('assert/strict');
const {JSDOM} = require('jsdom');
(async () => {
    const html = fs.readFileSync(0, 'utf8');
    const dom = new JSDOM(html, {url: 'http://localhost/subscriptions/90', runScripts: 'outside-only'});
    const w = dom.window;
    let audio;
    w.Audio = class extends w.EventTarget {
        constructor() { super(); audio = this; this.paused = true; this.currentTime = 0; this.duration = NaN; }
        async play() {
            if (this.fail) throw new Error('unavailable');
            this.paused = false;
            this.dispatchEvent(new w.Event('play'));
        }
        pause() { this.paused = true; this.dispatchEvent(new w.Event('pause')); }
    };
    w.eval(fs.readFileSync('app/web/static/js/episode-player.js', 'utf8'));
    const button = w.document.querySelector('[data-episode-play]');
    const controls = button.parentElement.querySelector('[data-episode-progress]');
    const seek = controls.querySelector('input');
    const tick = () => new Promise(resolve => setImmediate(resolve));
    button.click(); await tick();
    assert.equal(audio.src, '/episodes/90/audio');
    assert.equal(button.getAttribute('aria-label'), 'Pause episode');
    assert.equal(controls.hidden, false);
    assert.equal(seek.disabled, true);
    audio.duration = 120; audio.currentTime = 30;
    audio.dispatchEvent(new w.Event('loadedmetadata'));
    assert.equal(seek.value, '25');
    assert.equal(controls.querySelector('[data-episode-time]').textContent, '0:30 / 2:00');
    seek.value = '50'; seek.dispatchEvent(new w.Event('input', {bubbles: true}));
    assert.equal(audio.currentTime, 60);
    button.click(); await tick();
    assert.equal(audio.paused, true);
    assert.equal(button.getAttribute('aria-label'), 'Play episode');
    button.click(); await tick();
    assert.equal(audio.currentTime, 60);
    const second = button.closest('.episode-card').cloneNode(true);
    second.querySelector('[data-episode-play]').dataset.episodePlay = '91';
    second.querySelector('[data-episode-play]').dataset.audioUrl = '/episodes/91/audio';
    w.document.getElementById('episodes-grid').append(second);
    second.querySelector('[data-episode-play]').click(); await tick();
    assert.equal(audio.src, '/episodes/91/audio');
    assert.equal(button.getAttribute('aria-label'), 'Play episode');
    assert.equal(controls.hidden, true);
    audio.ended = true; audio.paused = true; audio.dispatchEvent(new w.Event('ended'));
    assert.equal(second.querySelector('[data-episode-play]').getAttribute('aria-label'), 'Play episode');
    audio.fail = true;
    second.querySelector('[data-episode-play]').click(); await tick();
    assert(second.querySelector('[data-episode-error]').textContent.includes('Unable to play'));
    // Search results can replace cards without losing the active control state.
    const replacement = second.cloneNode(true);
    second.replaceWith(replacement); await tick();
    assert.equal(replacement.querySelector('[data-episode-progress]').hidden, false);
    dom.window.close();
})().catch(error => { console.error(error); process.exit(1); });
