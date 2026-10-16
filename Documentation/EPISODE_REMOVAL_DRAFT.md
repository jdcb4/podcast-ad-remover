# Draft episode classification and removal specification

Date: 16 October 2026. Status: design draft for future work; backend approach and
open decisions are not yet selected. No implementation or deployment is included.

Episode Removal lets a podcast's owner choose which kinds of whole episodes appear
in the replacement feed. It is separate from Content Removal, which cuts ads and
other selected segments from episodes that are kept. Classification describes the
episode; settings determine whether to include it.

## Requested settings

Add an **Episode Removal** tab in per-podcast settings, with a master on/off toggle.
When enabled, show toggles for the episode types to remove. Add the corresponding
**Episode Removal** section under **Podcast defaults**, with the same master toggle
and category controls.

The initial taxonomy has twelve categories. Keep main episodes, crossovers, bonus
episodes, minisodes, mailbags, aftershows, live recordings and announcements.
Remove feed drops/cross-promotions, trailers, paywall teasers and reruns when those
category preferences are enabled. The complete supplied definitions, examples and
duration signals are preserved below. In particular, housekeeping announcements
are kept; a substantive crossover is different from an inserted episode of another
show, and a complete bonus is different from a truncated premium preview.

## Proposed settings behavior

Follow existing settings inheritance: a podcast inherits the Episode Removal group
from Podcast defaults unless its owner selects an override. Show effective inherited
values and preserve stored overrides when switching inheritance on and off.
Use one consistent meaning for category toggles: **Remove [episode type]**, on means
exclude and off means keep. Show all twelve types when the master toggle is enabled.
Disable category controls when it is off, retaining their saved selections.

Start the master toggle off for fresh installs and upgrades, pending confirmation.
When enabled for the first time, preselect removal for the four EXCLUDE categories
and keep the eight INCLUDE categories. This is an initial preset, not an immutable
mapping: classification and the user's category choices remain separate.

Proposed episode presentation: show the category, a short evidence-based explanation,
whether it was excluded by settings and a manual keep/restore override. Uncertain
classifications stay included. Existing deliberate ignore/delete decisions remain
separate from automatic filtering. Exclusion should hide an episode from generated
feeds and skip processing; it should not erase its database history or published
audio as a side effect. Retention remains a separate policy.

## Current implementation constraints

The current RSS adapter returns episode entries. Feed parsing retains title,
description/show notes, duration, GUID, publication date, enclosure URL and file size.
It does not currently retain RSS episode-type hints in these entries, or return a
feed context bundle with discovery. Source: `app/core/feed.py`, `app/core/sources.py`.

Processing currently downloads audio, transcribes it and then calls Complete
Timeline classification. That call supplies podcast name, episode title, publication
date, measured duration and the full numbered transcript/gap timeline. It does not
currently supply episode show notes, feed description, host roster or a duration
baseline. Source: `app/core/processor.py`, `app/core/timeline.py`.

Episode classification would therefore need either a new stage before download,
an extension of the existing structured timeline response, or both. The existing
provider timeout and durable request budget must cover any added calls. Classification
settings, prompt/schema/model and evidence inputs need reproducible job snapshots
and cache keys, following the existing processing model.

## Evidence the classifier needs

| Evidence | What it helps determine | Limitations |
| --- | --- | --- |
| Episode title and complete bounded show notes | Explicit trailer/feed-drop/preview/rerun wording and descriptions of collaboration | Promotional boilerplate and ordinary membership links occur on full episodes too |
| RSS duration and episode-type hint, when available | Short previews, regular releases, declared trailers or bonuses | Missing/zero duration is unknown; publisher hints and duration alone are insufficient |
| Feed title, description and known host names | Whether the episode is about this show or another show, and whether named hosts belong to this feed | Host names may be absent; inferred names must not be treated as verified identity |
| A bounded sample of recent titles, descriptions and valid durations | Typical formats, recurring minisodes, season labels and approximate duration baseline | A recent window may contain specials, reruns or a new format; it is context, not ground truth |
| Full transcript, when available | Whether there is substantive content, an invitation to hear the rest elsewhere, or only an introduction to another show | Transcription costs time and compute; text does not reliably identify every speaker or prove prior publication |
| Historical episode identity and fingerprints | Supporting evidence of a rerun or unchanged reupload | Similar topics/titles do not prove duplicates, and changed ads/edits can change audio fingerprints |

The supplied duration ranges are signals, not hard exclusion thresholds. A very
short episode can be a legitimate announcement or minisode. A long feed drop can
look like a normal episode by duration. A Patreon link does not itself make a
complete episode a paywall teaser. “Revisited” can mean a new discussion rather
than a rerun. “Best Of” can be a newly edited compilation. A brief host introduction
does not turn an inserted episode into a substantive crossover.

Start with a bounded feed context sample, for example up to twenty recent entries
and available declared host information. Strip markup and bound text sizes while
preserving meaningful text; treat feed/show-note/transcript text as untrusted source
data. Do not send signed enclosure URLs, private feed credentials or account secrets
to the LLM. Do not fetch other websites automatically to fill gaps. Missing context
should reduce certainty rather than encourage invented feed history or host facts.

## Backend options

| Option | Processing order | Advantages | Tradeoffs |
| --- | --- | --- | --- |
| A Metadata only | Eligible queued entry → classify title/show notes/duration/feed context → exclude or process normally | Can avoid both download and transcription; compact model input | Ambiguous collaboration, premium completeness and undeclared reruns may remain unresolved; adds a call for kept episodes |
| B Transcript classification | Download → transcribe → classify episode type alongside Complete Timeline → exclude or finish processing | Most audio-content evidence; potentially shares the existing analysis request | Saves no download/transcription work; expands schema/prompt and uses long transcript input even for obvious trailers |
| C Conservative hybrid | Metadata classification → exclude only clear selected categories → ambiguous cases continue to transcription and optional timeline classification | Avoids obvious unwanted work while retaining a deeper path for ambiguous cases | Two stages, caches and possibly two calls per ambiguous episode; needs an explicit escalation policy |

### Option A

Run one structured metadata classification before audio acquisition. Keep unknown or
mixed cases and send them through the current pipeline. A feed title, description,
available host identities and a bounded recent sample are important additions to
episode metadata; the title alone is too weak for general filtering. This option is
the smallest backend starting point, but cannot promise detection of all twelve
categories from metadata alone.

### Option B

Add a separate episode-type result to the existing timeline schema and pass the
missing metadata/feed context alongside the transcript. Classifying all segments
even when the episode will be excluded is simpler to validate but potentially wastes
output. An alternative short excluded-episode response would need an explicit schema
branch and validator; it must not masquerade as a complete valid timeline.
Transcript evidence improves some decisions but does not establish speaker identity
or historical duplication by itself. This option fits cases where avoiding mistaken
exclusion matters more than saving transcription work.

### Option C

Recommended target design, subject to evaluation: classify metadata first, cache the
result and exclude only clear evidence-backed cases in selected removal categories.
Keep ambiguous episodes; if deeper classification is enabled, let the existing
transcript stage resolve them before publication. The metadata stage should not
create a second provider/model fallback cascade. Escalation uses the configured
analysis provider/model and shares normal budgets and retry handling.

A practical first implementation could ship Option A conservatively, measure misses
and false exclusions, then add transcript escalation where it materially helps.
Do not automatically transcribe the entire feed merely to improve categorization.
If filtering is off, skip additional classification work. Keep paid calls out of the
feed-polling loop: persist eligible entries and run classification as bounded durable
work. Cache reuse across routine polls avoids repeatedly paying to classify unchanged
entries; changing removal preferences should apply cached labels without a fresh call.

## Proposed classification contract

Return a validated category, an evidence sufficiency/uncertainty result, a concise
reason and references to the relevant metadata fields or transcript items. Keep the
include/exclude decision in deterministic application code. Store provider/model,
prompt/schema versions, input/context fingerprints and stage provenance. Do not use
an uncalibrated model confidence percentage as the sole basis for exclusion.

Use an internal unknown result when evidence is insufficient; it is always kept and
does not add a thirteenth user-facing removal toggle. Invalid output, a timeout or
an unavailable provider must never silently exclude the episode. Decide whether such
operational failures keep or hold processing using existing job semantics before
implementation. Classification caching and manual overrides must respect ownership,
fenced job claims and cancellation. Any schema additions require compatible migration
and rollback/backup notes.

Some categories overlap: an episode can be live, bonus and a crossover, or a rerun
of any of those. Decide between a primary category with independent flags, or an
explicit precedence rule, before making toggle behavior contractual. In particular,
define what happens when a rerun is selected for removal but live episodes are kept.

## Open decisions

- Backend option and whether transcript escalation is automatic, optional or deferred.
- Whether to start with RSS only; YouTube has different metadata and discovery behavior.
- Master-off defaults, inheritance and whether per-user filtering is a later feature;
  current proposed controls operate at the shared podcast level.
- Whether the latest-N processing window counts retained episodes or all source
  entries. Recommend counting retained episodes, with bounded scanning, so a trailer
  does not occupy the only download slot; never turn that into an unbounded archive scan.
- Manual download/keep overrides, archive processing, and whether toggling preferences
  affects already-published episodes. Propose new processing only until the user
  explicitly requests a reviewed historical refresh.
- Category overlap resolution, mixed episodes and treatment of newly edited compilations.
- How many feed-context examples to retain, how a baseline is calculated, and how
  rebrands/host changes or edited show notes invalidate cached results.

## Evaluation before implementation is accepted

Build a labelled fixture set covering all twelve categories and ambiguous pairs:
feed drop/crossover, bonus/paywall preview, trailer/announcement, minisode/main and
rerun/new revisitation. Include sparse metadata, missing duration, ordinary paid
membership links, host introductions, overlapping categories and multilingual titles.
Compare metadata-only and transcript-assisted decisions on the same episodes.
Track false exclusions separately from missed exclusions, unknowns, request counts,
processing time and avoided download/transcription work. Conservative handling must
keep normal episodes when the evidence is unclear.

Exercise disabled settings, inheritance/override restoration, manual keeps, unchanged
classification reuse, failure/retry budgets and cancellation. Verify no accidental
media deletion, episode/job creation outside the selected window, or automatic changes
to existing publications. Review settings in desktop/mobile layouts with clear
distinction between whole-episode exclusion and segment removal.

## Supplied taxonomy

The following taxonomy preserves Joe's supplied category descriptions, detection
signals, initial actions and duration patterns. Those metadata signals describe the
initial proposal; backend alternatives above remain options for future selection.

### Podcast Episode Classification & Filtering Taxonomy

Use the following categories, descriptions, and text/duration signals to classify podcast episodes based solely on their title, description/show notes, and duration.

#### 1. INCLUDE (Keep in Feed)

##### Main Episode (`main_episode`)
* **Action:** INCLUDE
* **Description:** The core, standard release of the podcast, including multi-part series releases.
* **Detection Signals:** Duration aligns with the show's standard baseline length. The title focuses directly on the episode topic, guest, or sequential numbering (e.g., "Part 1", "Ep. 42") without promotional prefixes.

##### Crossover Collaboration (`crossover_episode`)
* **Action:** INCLUDE
* **Description:** A joint episode where the hosts of this podcast actively collaborate, co-host, or converse with the hosts of another podcast.
* **Detection Signals:** Titles or descriptions contain collaborative phrasing like "Crossover", "feat. [Host/Show]", "with [Other Show Name]", or "Special Joint Episode". The show notes confirm the primary hosts of *this* feed are active participants in the recording rather than just introducing another show's audio.

##### Bonus Episode (`bonus`)
* **Action:** INCLUDE
* **Description:** Full extra content released outside the regular schedule, such as extended cut interviews, behind-the-scenes chats, or tangential deep dives.
* **Detection Signals:** Titles or descriptions include "Bonus:", "Extra:", "Extended Cut", or "Unreleased", and the description does *not* end with a paywall prompt to hear the rest of the audio elsewhere.

##### Minisode / Short (`minisode`)
* **Action:** INCLUDE
* **Description:** A complete, self-contained, bite-sized version of the regular show format designed for quick listening.
* **Detection Signals:** Shorter than main episodes (typically 5–20 minutes). Titles contain "Minisode", "Mini:", "Quick Take", "Short", or a recurring short-segment title.

##### Q&A / Mailbag (`mailbag`)
* **Action:** INCLUDE
* **Description:** Episodes dedicated to answering listener-submitted questions, reading reviews, or addressing community feedback.
* **Detection Signals:** Titles and descriptions feature "Q&A", "AMA", "Ask Me Anything", "Mailbag", "Listener Questions", or references to voicemail/email submissions.

##### Aftershow / Post-Mortem (`aftershow`)
* **Action:** INCLUDE
* **Description:** Casual debriefs where hosts discuss the main episode, reflect on an interview, or chat loosely after the main recording.
* **Detection Signals:** Titles include "Aftershow", "Post-Show", "Debrief", "Cooldown", "Post-Mortem", or "More with [Guest]".

##### Live Recording (`live_episode`)
* **Action:** INCLUDE
* **Description:** An episode recorded in front of a live audience at a festival, theater, or virtual event.
* **Detection Signals:** Titles or descriptions explicitly mention "Live from [City/Venue]", "Live Show", or "Recorded live at [Event]".

##### Housekeeping / Feed Update (`announcement`)
* **Action:** INCLUDE
* **Description:** Short audio notes directly from the hosts regarding schedule changes, hiatus announcements, live tour dates, or feed migrations.
* **Detection Signals:** Very short duration (typically 1–5 minutes). Titles include "Announcement", "Important Update", "Where we've been", "Schedule Update", or "Live Show Info".

---

#### 2. EXCLUDE (Remove from Feed)

##### Feed Drop / Cross-Promotion (`cross_promo`)
* **Action:** EXCLUDE
* **Description:** An episode from an entirely different podcast inserted into the feed for marketing purposes, where the primary hosts of this podcast are not participating in the main content.
* **Detection Signals:** Titles begin with prefixes like "Introducing:", "Presenting:", "[Network] Presents:", "From our friends at", "Feed Drop:", or "Check out:". The description primarily pitches a *different* podcast and provides links to subscribe to that separate feed.

##### Show or Season Trailer (`trailer`)
* **Action:** EXCLUDE
* **Description:** A short promotional preview advertising the podcast itself or an upcoming season.
* **Detection Signals:** Duration is typically under 5 minutes. Titles and descriptions include keywords like "Trailer", "Season X Teaser", "Coming Soon", or "Welcome to [Show Name]".

##### Subscription Teaser / Paywall Preview (`paywall_teaser`)
* **Action:** EXCLUDE
* **Description:** A truncated snippet or preview of a premium episode meant to drive listeners to a paid membership tier.
* **Detection Signals:** Duration is noticeably shorter than a standard episode (often 3–10 minutes). Titles include "[Preview]", "[Teaser]", "Patreon Preview", or "Subscriber-Only". The description contains a call-to-action to unlock or listen to the full episode via Patreon, Supercast, Apple Subscriptions, or another paid link.

##### Rerun / Rebroadcast (`rerun`)
* **Action:** EXCLUDE
* **Description:** A previously aired episode re-uploaded to fill a scheduling gap or highlight archival content.
* **Detection Signals:** Titles contain "[Rerun]", "Rebroadcast", "Classic:", "From the Vault", "Encore", "Best Of", "Flashback", or "Revisited". Descriptions include phrases like "Originally aired on [Date]", "In case you missed it", or "While we are on break, enjoy this classic episode".

---

#### Summary Reference Table

| Category ID | Action | Primary Title & Description Signals | Duration Pattern |
| :--- | :--- | :--- | :--- |
| `main_episode` | **INCLUDE** | Standard topical/guest title, sequential numbers ("Ep. 12", "Part 1") | 100% of baseline |
| `crossover_episode` | **INCLUDE** | "Crossover", "feat. [Host]", joint host discussion in show notes | 80–120% of baseline |
| `bonus` | **INCLUDE** | "Bonus", "Extra", "Extended Cut" (without paywall links) | 30–100% of baseline |
| `minisode` | **INCLUDE** | "Minisode", "Short", "Mini", "Quick Take" | 5–20 mins |
| `mailbag` | **INCLUDE** | "Q&A", "Mailbag", "AMA", "Listener Questions" | 30–100% of baseline |
| `aftershow` | **INCLUDE** | "Aftershow", "Post-Mortem", "Debrief", "Cooldown" | 20–80% of baseline |
| `live_episode` | **INCLUDE** | "Live from...", "Recorded live at...", "Live Show" | 80–120% of baseline |
| `announcement` | **INCLUDE** | "Update", "Announcement", "Housekeeping", "Schedule Change" | 1–5 mins |
| `cross_promo` | **EXCLUDE** | "Introducing:", "Presenting:", "Feed Drop", links to another show | Varies |
| `trailer` | **EXCLUDE** | "Trailer", "Season Teaser", "Coming Soon" | Under 5 mins |
| `paywall_teaser` | **EXCLUDE** | "Preview", "Teaser", "Hear the full episode on Patreon/Substack" | 3–10 mins |
| `rerun` | **EXCLUDE** | "Rerun", "Rebroadcast", "Encore", "Vault", "Originally aired..." | 100% of baseline |
