---
name: Podcast Ad Remover
description: Compact podcast operations with violet actions and theme-aware surfaces.
colors:
  primary: "#7c3aed"
  primary-deep: "#6d28d9"
  link-dark: "#a78bfa"
  link-light: "#6d28d9"
  surface-base-dark: "rgb(10 10 15)"
  surface-dark: "rgb(18 18 26)"
  surface-elevated-dark: "rgb(26 26 37)"
  foreground-dark: "rgb(237 237 239)"
  surface-base-light: "rgb(246 247 251)"
  surface-light: "rgb(255 255 255)"
  surface-elevated-light: "rgb(237 240 247)"
  foreground-light: "rgb(24 32 49)"
typography:
  headline:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "24px"
    fontWeight: 650
    lineHeight: 1.2
  body:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "16px"
    lineHeight: 1.6
  label:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "14px"
  mobile-title:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "15px"
  configurator-body:
    fontFamily: "system-ui, sans-serif"
rounded:
  artwork: "6px"
  navigation: "8px"
  control: "12px"
  overlay: "16px"
spacing:
  compact: "8px"
  row: "12px"
  section: "16px"
  group: "24px"
components:
  button-primary:
    textColor: "#ffffff"
    rounded: "{rounded.control}"
    padding: "10px 20px"
    typography: "{typography.label}"
  button-secondary:
    rounded: "{rounded.control}"
    padding: "10px 20px"
    typography: "{typography.label}"
  input:
    rounded: "{rounded.control}"
    padding: "12px 16px"
  navigation:
    rounded: "{rounded.navigation}"
    padding: "10px 12px"
  setting-switch:
    backgroundColor: "{colors.primary}"
    width: "40px"
    height: "24px"
  card:
    rounded: "{rounded.control}"
    padding: "24px"
---

# Design System: Podcast Ad Remover

## Overview

**Creative North Star: "Operate"**

A compact workspace for finding podcasts and changing their processing settings. Content and actions lead; labels are concise, help appears when useful, and advanced controls sit behind disclosures. Violet identifies actions within quiet dark or light surfaces.

This is a code-led record of the current system, not a generated visual comp. The authoritative sources are `app/web/static/css/input.css`, its Tailwind configuration and compiled utilities, then `app/web/static/css/v2.css` loaded afterward by `base.html`. Settings patterns come from `_settings.html` and the admin templates. `PRODUCT.md` and `Documentation/V2_PROPOSAL.md` supply the accepted product direction; implemented styles supply the values.

**Key Characteristics:**

- Compact labeled controls and restrained helper text.
- One desktop sidebar and phone bottom navigation.
- Artwork, wrapping title and manage action on phone podcast rows.
- Violet actions and theme-aware neutral surfaces.

## Colors

The app uses electric violet against layered cool neutrals. The frontmatter captures the reusable core, not every utility color.

Primary and primary-deep form the existing diagonal primary-button gradient. Link colors adapt by theme. Surface triplets in the source are rendered as CSS rgb colors here; they are not independently chosen approximations. Base is the page canvas, surface is content, elevated is navigation and secondary controls, and foreground is primary text. The CSS variables remain the runtime authority for hover, secondary text, borders and semantic status colors.

**The Theme Rule.** Use semantic surface, text and status variables for new application content; verify both themes.

The standalone installation configurator deliberately has its own dark-only palette in `configurator/style.css`: canvas `#15151b`, controls `#22222d`, text `#e9e9f1` and filled actions `#8060c5`. It is an independent installation page with no app shell or theme switch. Do not silently substitute these values for app tokens. Its palette is implementation-specific rather than a new shared theme.

## Typography

The application loads Inter in `base.html`; the applied body utility resolves to Inter with system-ui and sans-serif fallbacks. The base CSS also includes platform-specific fallbacks. There is no newly introduced display face. Normal body copy uses the browser/Tailwind base size; compact settings labels use the smaller label role. Settings page titles use the headline role, while phone podcast titles use mobile-title and wrap freely.

The configurator intentionally uses system-ui without loading Inter. Its installation heading uses `clamp(2rem,5vw,2.9rem)`, line-height `1.15` and letter-spacing `-.04em`; this isolated heading treatment is not a shared app display token. Existing Tailwind display/fluid scale definitions are not evidence that new settings pages should use them.

## Layout

The desktop sidebar is fixed at 240px wide with 24px vertical and 16px horizontal padding. Main content offsets by that width, uses 28px by 32px padding and caps at 1600px; settings content caps at 960px. Sub-navigation expands within the same sidebar.

At 900px and below, the sidebar becomes a settings-and-account drawer; an app bar and fixed bottom navigation replace the persistent rail. The drawer omits the primary destinations already present below. Main padding becomes 18px by 16px with 88px plus the safe-area inset reserved beneath content. Bottom navigation actions have a 48px minimum height.

Settings rows have a 58px minimum height, 12px vertical padding and a fine bottom divider. Direct text/select controls occupy 22rem, capped at 58%; below 900px they occupy 55%. At 480px and below, those rows stack the label over a full-width control. Numeric controls and switches retain their compact inline relationship.

Phone podcast rows use 48px square artwork, a flexible wrapping title and a 44px square manage action. The configurator independently caps its column at 780px and stacks fields at 520px; these breakpoints serve its installation form, not app navigation.

**The Compact Control Rule.** Keep each setting visibly labeled, align repeated controls, and disclose advanced options instead of filling the page with helper prose.

## Elevation & Depth

Settings rely on thin separators and tonal surface changes. The incumbent component library still uses gradient cards, violet button glow and soft overlay shadows; do not describe the entire application as shadow-free. Primary button glow is `0 0 20px rgba(139, 92, 246, 0.3)`, increasing on hover. Cards lift on hover, while settings rows remain stationary.

Native v2 dialogs use a dim backdrop. The older confirmation wrapper adds blur and a nested elevated panel. Motion is short feedback rather than continuous decoration; v2 disables animations and transitions when reduced motion is requested. Exact shadow and transition values live in the sidecar.

## Shapes

Controls and cards use gently rounded corners; navigation is slightly tighter and artwork tighter again. Switch tracks are pill-shaped with a circular white thumb. Dividers separate settings without enclosing every row in a card. Native dialogs use the control radius; the nested confirmation panel uses the overlay radius.

## Components

**Buttons.** Primary actions use white text over the violet gradient, a 44px minimum height, medium label text and subtle press scaling. Secondary buttons use elevated surfaces and a border. Ghost actions are transparent until hover. Existing small variants have a 36px minimum height; do not apply those to new phone primary actions.

**Fields.** Inputs and selects use the base surface, primary text, a two-pixel control border and a violet focus border plus four-pixel translucent ring. A label names the setting; units remain next to numeric values. The shared macros preserve label/control association.

**Settings hierarchy.** A concise introduction explains each page using secondary text (14px, line-height 1.6, maximum 72ch). Section headings and legends are 17px and weight 650, with 22px above and 8px below. Contextual help is 13px with line-height 1.5 and a 75ch maximum. These additions retain the existing compact type scale rather than introducing display typography.

**Catalog selectors.** Model and voice controls expose a real select with an explicit manual-ID choice. Choosing manual entry reveals a labeled text field below the select. Their shared container is 22rem capped at 58%, with an 8px gap; at 480px it stacks below the row label at full width. Refresh and preview actions have adjacent status feedback.

**Switches.** A white thumb moves 16px between gray and violet tracks. The complete labeled row is the checkbox label, so the visible track is not the entire activation target.

Unavailable speech switches retain their saved preferences and show disabled controls at 0.45 opacity with secondary label text. Nearby help explains how to configure speech. Disabled appearance communicates availability, not a reset preference.

**Navigation.** Current destinations receive a violet wash and stronger weight. Inline stroke SVG icons accompany visible labels, with a 10px gap in the sidebar; phone bottom navigation stacks icons over labels with a 3px gap. Sidebar settings links remain subordinate to the main destinations. The mobile settings-and-account drawer has a 44px close action, dim clickable backdrop, contained keyboard focus and focus return to the menu trigger. The background shell becomes inert while it is open. A single RSS-marked Unified Feed action sits in the browsing toolbar instead of repeating feed actions in navigation.

**Disclosures.** Native details/summary groups retain keyboard behavior and reveal advanced settings in place. Summary rows are compact, separated by rules, with semibold labels.

**Dialogs.** Add and manage workflows use native dialogs with a close control and constrained scrolling. General v2 dialogs cap at 650px and 85dvh, with 24px padding. Use the native top layer for new focused workflows; existing legacy overlays are not a new reusable modal pattern.

**Cards and chips.** Existing desktop/episode cards retain their gradient surface, border and hover lift. Status chips are compact rounded pills with semantic colors. They communicate real state, not decorative claims.

## Do's and Don'ts

### Do:

- Do keep labels visible and advanced settings in disclosures.
- Do use semantic theme variables and check light and dark surfaces.
- Do preserve wrapping phone titles and usable action targets.
- Do use native dialogs for new focused workflows.

### Don't:

- Don't add a second permanent administration rail.
- Don't introduce invented status metrics or decorative live badges.
- Don't turn the standalone configurator palette into the app theme.
- Don't treat legacy display utilities or overlay wrappers as new shared patterns.

## October 2026 settings refinement

Keep the existing visual identity. Place Save at the top of each settings form; put immediate actions in labelled rows and report their outcome without saving unrelated drafts. Use aligned control columns, units in labels and mobile stacking. Model/voice refresh buttons are 44px icon controls beside selectors. Voice preview uses the unsaved selection. Prompt definitions stay expanded and grow with their contents; Restore default changes only the draft. GPU, retention/timing and system resource sections stay expanded.

Podcast settings starts collapsed with an accessible chevron. Its compact groups retain inheritance and effective-value controls. Owner, source replacement, whole-show batches and labelled deletion live inside it. Subscription actions share the authenticated dialog and remain visible without hover. Upgrade history is historical migration information, not a live status panel.

Podcast settings use four compact columns at 1200px and above, two below that, and one at 640px and below. GPU execution/setup details reuse label/value rows. Prompt editors have a 4px heading gap and resize to content, including after font loading.

## Podcast settings navigation (Option A, 2026-10-03)

The prior four-column layout is superseded by Processing, Downloads and Manage
tabs. Processing uses two compact columns plus a collapsed Advanced section.
Phones use one accordion level with only one section open. Inherited groups show
effective summaries and Customize; controls stay in the same form across tabs,
preserving drafts. Timing keeps the content-removal inheritance policy even in
Advanced. Save is shown only when processing preferences differ from the loaded
values. Owner, feed/archive and delete actions remain separate immediate actions
with their existing permissions and review safeguards.
