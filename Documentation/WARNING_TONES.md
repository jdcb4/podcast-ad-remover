# Removal warning tones

Use **Podcast Preferences > Global Subscription Settings > Removal Warning Tones** to enable
start, middle, or end cues independently, and preview/select a sound for each position.
All switches start off. Wooden notes is the initial selection; styles can be mixed.

- A beginning cue plays only when source content was removed before the first retained audio.
- An ending cue plays only when source content was removed after the last retained audio.
- One low cue plays at each interior removal seam. Overlapping or adjacent cuts create one seam.
- No cuts means no cues. An entirely removed episode still fails instead of publishing only tones.
- Optional spoken titles/summaries precede the beginning cue; the cue marks the start of retained podcast audio.

These settings are global for all podcasts and are read when processing begins. They apply to both
Legacy and Complete Timeline, including SponsorBlock cuts. Source timestamps and removal statistics
remain on the original timeline; the published duration includes inserted cues. Existing episodes
need reprocessing to receive changed sounds.

## Sound choices

| Style | Character | Beginning / ending |
| --- | --- | --- |
| Soft chime | Gentle pure tones | Three ascending / descending notes |
| Warm mellow | Lower, rounded tones | Three ascending / descending notes |
| Clear signal | Brighter, higher tones | Three ascending / descending notes |
| Gentle bell | Soft harmonic bell | Two ascending / descending notes |
| Sonar pair | Smooth pulsing tones | Two ascending / descending notes |
| Wooden notes | Short, damped harmonic taps | Two ascending / descending notes |

Every style includes one low middle-removal tone. Edge cues last approximately 0.47–0.70 seconds;
middle cues last 0.25 seconds. Peak amplitude is bounded around -18 dBFS with smoothed attack/release.
Actual perceived loudness depends on the source podcast and playback volume.

Listen in the settings page or open [the standalone sound gallery](WARNING_TONE_SAMPLES.html).

## Assets and implementation

The 18 original PCM WAV files live in `app/web/static/audio/warning-tones/` and ship with the app
under the repository's MIT license. They are 22.05 kHz mono, 16 bit. No third-party sound license,
network download, synthesis, model or TTS engine is needed during processing. FFmpeg resamples
and splices the existing files into its normal cutting pass.

`python -m app.core.warning_tones` regenerates the bundled assets during development only.
Commit both the generator source and resulting WAVs when changing sounds. Audio previews use those
same WAVs, with no autoplay and no JavaScript dependency.

## Migration and rollback

Migration `20260913_0018_warning_tones` adds six settings columns, preserving all existing media and
podcast settings. The normal migration runner creates a database backup under `/data/backups/`
before formal migrations. Older code ignores the added columns, so rolling back code leaves the
database readable. To restore exact previous settings, use the pre-migration backup following
[Recovery](RECOVERY.md); do not delete or reset `/data`. Model-default migration
`20260913_0017_model_defaults` changes only known prior defaults, once; customized model lists and
already frozen Complete Timeline job snapshots remain unchanged.

Migration `20260913_0019_wooden_tone_default` changes the former Soft chime defaults to Wooden notes once, preserving other selected styles and all enable switches. The same pre-migration backup/rollback procedure applies.
