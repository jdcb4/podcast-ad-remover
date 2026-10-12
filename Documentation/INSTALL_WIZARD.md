# Install with the web wizard

The [web setup wizard](https://jdcb4.github.io/podcast-ad-remover/) prepares a new Docker installation without requiring you to write a Compose file. It runs entirely in your browser, including from an offline ZIP. Inputs are not uploaded, tracked or persisted in browser storage.

## Stable installer

The homepage opens the stable wizard directly, with no channel selector. Existing
`/stable/` links continue to work; `/dev/` redirects to the stable homepage and its
old Dev downloads are removed on publication. Published assets display the exact
production image and revision. The source checkout defaults to `:latest`.

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

## Download one HTML file

The hosted wizard offers **Download single HTML wizard**. You can also
[download the stable repository copy](https://github.com/jdcb4/podcast-ad-remover/raw/refs/heads/gh-pages/stable/install.html),
save it as an HTML file and open it in your browser. No Python, local server or
other files are needed. Published copies display their exact release
image/revision; the source checkout's copy uses the stable `:latest` image. Always check
that metadata before installing. The file generates configuration offline; Docker
still needs access to the image registry when you install the application.

Scripts, styling and release metadata are embedded. CSP permits only those exact
script/style hashes and blocks network requests. Credentials stay in memory until
you download `install.env`; use **Clear credentials & output** when finished.
Navigation also clears the temporary state. The existing offline ZIP remains
available with separate source assets and the portable agent package.

## Test the current checkout

Build a local image:

```bash
docker build -t podcast-ad-remover:v2-local .
```

Generate the install files, then replace the output image with `podcast-ad-remover:v2-local` before running. Use an isolated new volume/project or host directory and an unused port. Do not use your live data. For upgrades, rehearse a backup copy instead.

## Maintainer publication checklist

1. Restrict the `github-pages` environment to `main`. The dispatch helper uses `main`, and the workflow skips other refs. Stable revisions must belong to `main`, match the package version and have an existing immutable Docker image.
2. After authorized stable image publication, dispatch `publish-configurator.yml` from `main` with its full commit SHA and immutable image tag. Dev Docker publication does not dispatch Pages. Retries retain the intended revision/image.
3. Check the homepage, `/stable/`, retired `/dev/` redirect, `release.js`, single-file `install.html`, offline ZIP and agent ZIP. Verify generated install output and displayed metadata against the image. Rebuild and commit repository `install.html` whenever source assets change.
4. Confirm live success before announcement. Installer-only maintenance does not publish a new Docker image.
