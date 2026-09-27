# Cut tones

Enable one **Insert tone at content cuts** switch in Podcast defaults. Wooden notes is fixed. The switch inserts cues at every applicable beginning, interior and ending cut; it adds nothing when no content is removed. Spoken intros/summaries precede the retained-audio cue. Entirely removed episodes are skipped.

V2 enables this switch if any old position switch was enabled. Mixed settings therefore enable cues at every applicable position; the System upgrade report discloses this. Old columns remain only for recovery, and published audio changes only on reprocessing.

Fresh databases enable cut tones. Existing installations use the migration rule above; no other fresh-install default overrides their saved tone intent. This is a removal indicator, not a warning/alarm system.

## Fixed sound

Wooden notes uses short, damped harmonic taps: two ascending notes at the beginning,
two descending notes at the end, and one low note at an interior removal.
Edge cues last approximately 0.47 seconds and middle cues last 0.25 seconds.
The standalone [personal audition gallery](WARNING_TONE_SAMPLES.html) retains the original
alternatives for development reference only; these are not settings in the app.

## Assets and implementation

The 18 original PCM WAV files live in `app/web/static/audio/warning-tones/` and ship with the app
under the repository's MIT license. They are 22.05 kHz mono, 16 bit. No third-party sound license,
network download, synthesis, model or TTS engine is needed during processing. FFmpeg resamples
and splices the existing files into its normal cutting pass.

`python -m app.core.warning_tones` regenerates the bundled assets during development only.
Commit both the generator source and resulting WAVs when changing sounds. Audio previews use those
same WAVs, with no autoplay and no JavaScript dependency.

## Migration and rollback

See [V2_IMPLEMENTATION.md](V2_IMPLEMENTATION.md). Restore the matching pre-upgrade database and image for rollback; preserve `/data` and original media.
