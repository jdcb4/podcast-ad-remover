# PAR agent skill

The portable `podcast-ad-remover` skill teaches an HTTP-capable agent to browse PAR, read episode artifacts, import feeds, adjust podcast settings and manage processing using the supported v1 API. It is plain Markdown plus optional agent UI metadata; no MCP server, executable installer, or bundled credentials are required.

## Get it

Download **PAR agent skill** from your release channel's install configurator. The Pages deployment workflow packages the skill and the exact API guide from the published revision. Dev and stable packages remain separate. This becomes available when that revision is published; a local build alone does not update Pages.

From a source checkout, build the same archive:

```bash
python scripts/package_agent_skill.py dist/podcast-ad-remover-skill.zip
```

Unzip it and give your agent the `podcast-ad-remover` folder. For Codex, place that folder under `$CODEX_HOME/skills` (usually `~/.codex/skills`); consult your agent's own skill loader for other runtimes. Alternatively, attach `SKILL.md` and the files in `references/` as instructions/reference material to an HTTP-capable agent. A chat-only agent with no HTTP/network tool cannot operate PAR just by reading the skill.

## Connect it

1. In PAR, open **Settings → Users & access → API tokens**, enable the API, and create a token linked to the intended user. Grant only needed scopes; `admin` does not imply other scopes.
2. Supply the base URL as `PAR_BASE_URL` and the token as `PAR_API_TOKEN` through your agent runtime's secret/environment configuration. Never put the real token in the skill, a prompt, a checked-in env file, or a URL.
3. Ask the agent, for example: “Use the podcast-ad-remover skill to list my instance's podcasts,” or “Preview this Pocket Casts OPML file and import the valid new subscriptions.”

The skill checks the live OpenAPI contract, distinguishes global library from ownership, previews imports, respects rate limits, and explains artifact deletion and processing costs when relevant to the requested action. It does not authorize actions beyond the user's request. It deliberately excludes Docker/server administration and internal dashboard routes.

## Maintenance and validation

`Documentation/API.md` is the canonical human reference; the ZIP embeds it at build time to avoid a second maintained copy. The live `/api/v1/openapi.json` supplies installed schemas and `x-required-scopes`. API contract tests compare documented operations with the actual router, check scope metadata, validate the skill package and exercise import behavior without live feeds or paid services.

When modifying API behavior, update the API guide and skill operation notes in the same change, run `npm run verify`, and regenerate the archive. The package is intended for PAR V2; agents should use the installed schema to detect unsupported operations on older instances. V2 is not declared published by this documentation update. The bundle also carries the linked upgrade/recovery references so local API documentation links remain useful offline.
