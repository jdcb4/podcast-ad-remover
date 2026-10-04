# Install with the web wizard

The [web setup wizard](https://jdcb4.github.io/podcast-ad-remover/) prepares a new Docker installation without requiring you to write a Compose file. It runs entirely in your browser, including from an offline ZIP. Inputs are not uploaded, tracked or persisted in browser storage.

## Availability during the 2.0 preview

On 4 October 2026 the Pages homepage still shows the old Local LLM Ad Detection Evaluation report, and the Dev wizard path returns 404. The first Pages deployment was rejected because its environment still permits `master` and an old experimental branch, not `dev`. Use the local preview until publication succeeds. See the [readiness review](V2_LAUNCH_REVIEW.md) for evidence and remaining launch work.

Once published, the landing page links to available channels:

- [Stable installer](https://jdcb4.github.io/podcast-ad-remover/stable/): an explicitly published production version.
- [Dev installer](https://jdcb4.github.io/podcast-ad-remover/dev/): an immutable published Dev revision.

Check the image and revision displayed on the page. Neither channel means that every source commit has been published. The source preview uses the rolling `:dev` tag, not an exact build of your checkout.

## Prepare a new installation

1. Install Docker. For Compose output, use Docker Compose 2.30 or newer (the env file uses `format: raw`).
2. Select Compose or Docker run and your command shell. Choose a port and persistent storage. A named volume is suitable for a fresh install; a host directory must be an absolute path on the Docker host.
3. Set Application URL to the server address your podcast player can reach. `localhost` works only on the same machine. Include the chosen port when connecting directly.
4. Enable the HTTPS proxy option only if your container is reachable exclusively through your trusted reverse proxy. GPU and separate media storage are optional; see [CUDA](CUDA.md) and [Storage](STORAGE.md).
5. Leave credentials blank to configure them later, or enter a provider key. Environment keys override saved keys. No speech provider is automatically enabled.
6. Select **Build configuration**, review the output, and download both the install file and `install.env` into the same directory. Keep `install.env` private; it contains the generated session secret and any provider key.
7. Run the displayed command. Open the Application URL and complete or dismiss the in-app setup wizard. It can be rerun from System.

The in-app wizard is a separate five-step flow: application address, analysis provider/model, transcription, removal/retention, then review and apply. Changes stay in a temporary server draft until applied; cancellation discards them. Choose a model with native structured-output support. Configure optional speech later in Settings → Voice.

## Existing installations

Use [V2_UPGRADE.md](V2_UPGRADE.md), not a newly generated install file as a drop-in replacement. The browser wizard creates a fresh session secret and default volume configuration; it cannot discover your existing installation. Preserve your current secret, exact volume/bind mount, Compose project identity and image recovery point. Changing Compose folders/project names can select a different named volume. Switching between Compose and Docker run can also select different volume names.

Drain jobs and rehearse against backups before upgrading. Do not attach two running instances to the same writable data. Image-only rollback after database migration is unsupported.

## Local and offline preview

From the repository root, run `python scripts/build_configurator.py`, then open `configurator/index.html`. This also builds `offline.zip` and the portable agent package. Alternatively:

```bash
python -m http.server 8778 --bind 127.0.0.1 --directory configurator
```

Open `http://127.0.0.1:8778`. Serve only the configurator folder, not your repository or data directory. The offline ZIP can be extracted and its `index.html` opened without a server. Use **Clear credentials & output** when finished; a page navigation also clears temporary configuration state.

## Test the current checkout

Build a local image:

```bash
docker build -t podcast-ad-remover:v2-local .
```

Generate the install files, then replace the output image with `podcast-ad-remover:v2-local` before running. Use an isolated new volume/project or host directory and an unused port. Do not use your live data. For upgrades, rehearse a backup copy instead.

## Maintainer publication checklist

1. Review the GitHub `github-pages` environment branch rules. The dispatch helper runs the workflow from `dev` for both channels; allow that intended branch explicitly. Retain the workflow's checks that stable revisions belong to `main`, Dev revisions belong to `dev`, and immutable images exist. Do not disable environment protection wholesale.
2. After authorized image publication, dispatch `publish-configurator.yml` with its full commit SHA, channel and immutable image tag. A workflow retry must retain the intended revision/image; do not substitute a rolling tag. Preparing docs does not authorize publication or permission changes.
3. Check the landing page, channel page, `release.js`, offline ZIP and agent ZIP. Verify the other channel was preserved. Generate and parse install output, and verify displayed metadata against the published image.
4. Replace the temporary unavailable notice in README and this guide only after live success. Before stable 2.0, also finish Joe's rationale review and the [release checklist](VERSIONING.md).
