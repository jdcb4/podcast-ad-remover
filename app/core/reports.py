"""Self-contained, escaped ad reports. All feed/model text is untrusted."""
import html


def _timeline_sections(analysis: dict, policy: dict) -> str:
    from app.core.timeline import LABEL_NAMES
    escape = lambda value: html.escape(str(value))
    parts = ['<h3>Complete timeline classification</h3>',
             '<p>Model categories are recorded before user preferences and the short-island rule. Gaps are contextual evidence, not measured silence.</p>',
             f'<p>Provider: {escape(analysis.get("provider"))}; model: {escape(analysis.get("model"))}; output: {escape(analysis.get("output_mode"))}; prompt: {escape(analysis.get("prompt_version"))}</p>',
             f'<h3>Episode summary</h3><p>{escape(analysis.get("summary") or analysis.get("summary_error") or "Summary unavailable")}</p>',
             f'<h3>Cut preferences</h3><p>Remove: {escape(", ".join(LABEL_NAMES.get(k, k) for k in policy["remove_categories"]) or "Nothing")}. '
             f'Short-island threshold: {policy["minimum_retained_seconds"]:g}s (0 disables).</p>',
             f'<p>Extra cuts from the short-island rule: {policy["island_seconds"]:.2f}s.</p>']
    for island in policy['island_cuts']:
        parts.append(f'<p>{island["start"]:.2f}–{island["end"]:.2f}s: {escape(island["reason"])}</p>')
    if analysis.get('normalization_notes'):
        parts.append('<details><summary>Transcript timestamp normalization</summary><p>' + escape(analysis['normalization_notes']) + '</p></details>')
    units = analysis['timeline']
    for row in analysis['segments']:
        removed = sum(max(0, min(row['end'], cut['end']) - max(row['start'], cut['start'])) for cut in policy['segments'])
        decision = 'Keep' if removed < 1e-7 else 'Remove' if removed >= row['end'] - row['start'] - 1e-7 else 'Partly remove'
        text = row.get('text') or ''
        if not text:
            before = next((u['text'] for u in reversed(units[:row['first_id'] - 1]) if u['text']), 'Start of episode')
            after = next((u['text'] for u in units[row['last_id']:] if u['text']), 'End of episode')
            text = f'GAP: no text captured. Before: {before} After: {after}'
        parts.append(f'<details class="segment"><summary><strong>{row["start"]:.2f}–{row["end"]:.2f}s · '
                     f'{escape(LABEL_NAMES.get(row["label"], row["label"]))} · {decision}</strong></summary>'
                     f'<p>{escape(row["reason"])}</p><p class="transcript-text">{escape(text)}</p></details>')
    return ''.join(parts)


def render_ad_report(ep, ad_segments: list[dict], *, analysis: dict | None = None, edit_policy: dict | None = None) -> str:
    rows_html = ""
    for s in ad_segments:
        sponsorblock_evidence = []
        for evidence in s.get("evidence", []):
            if evidence.get("source") != "sponsorblock":
                continue
            sponsorblock_evidence.append(
                "<li>"
                f"category={html.escape(str(evidence.get('category') or 'unknown'))}; "
                f"UUID={html.escape(str(evidence.get('uuid') or 'unknown'))}; "
                f"votes={html.escape(str(evidence.get('votes')))}; "
                f"locked={html.escape(str(evidence.get('locked')))}; "
                f"action={html.escape(str(evidence.get('action_type') or 'skip'))}"
                "</li>"
            )
        evidence_html = (
            '<ul class="evidence">' + "".join(sponsorblock_evidence) + "</ul>"
            if sponsorblock_evidence
            else ""
        )
        rows_html += f"""
        <div class="segment">
            <div class="flex justify-between">
                <strong>{html.escape(str(s['start']))}s - {html.escape(str(s['end']))}s</strong>
                <span class="badge">{html.escape(str(s.get('label', 'Ad')))} · {html.escape(str(', '.join(s.get('sources', ['llm']))))}</span>
            </div>
            <p class="reason">{html.escape(str(s.get('reason', 'No reason provided')))}</p>
            {evidence_html}
            <div class="transcript-text">
                "{html.escape(str(s.get('text', 'No text extracted')))}"
            </div>
        </div>
        """

    html_content = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Ad Report: {html.escape(str(ep.title))}</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                font-family: 'Inter', sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 2rem 1rem;
                background: #0a0a0f;
                color: #fafafa;
                line-height: 1.6;
                overflow-wrap: anywhere;
            }}
            h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; }}
            h1 {{ font-size: 2rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #a78bfa, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
            h2 {{ font-size: 1.5rem; margin-bottom: 1rem; color: #fafafa; }}
            h3 {{ font-size: 1.125rem; margin: 1.5rem 0 1rem 0; color: #a1a1aa; }}
            .meta {{ color: #52525b; font-size: 0.85em; margin-bottom: 1.5rem; font-family: monospace; }}
            /* Scrollbar */
            ::-webkit-scrollbar {{ width: 8px; }}
            ::-webkit-scrollbar-track {{ background: #0a0a0f; }}
            ::-webkit-scrollbar-thumb {{ background: #22222f; border-radius: 4px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: #3f3f46; }}

            .segment {{
                background: #1a1a25;
                padding: 1.25rem;
                margin: 1rem 0;
                border-left: 4px solid #8b5cf6;
                border-radius: 0.75rem;
                border: 1px solid rgba(255,255,255,0.08);
            }}
            .badge {{
                background: rgba(139,92,246,0.15);
                color: #a78bfa;
                padding: 0.25rem 0.75rem;
                border-radius: 999px;
                font-size: 0.75em;
                font-weight: 600;
                border: 1px solid rgba(139,92,246,0.2);
            }}
            .badge.intro {{ background: rgba(52,211,153,0.15); color: #34d399; border-color: rgba(52,211,153,0.2); }}
            .badge.outro {{ background: rgba(251,191,36,0.15); color: #fbbf24; border-color: rgba(251,191,36,0.2); }}
            .flex {{ display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }}
            .transcript-text {{
                background: rgba(255,255,255,0.03);
                padding: 0.75rem 1rem;
                border-radius: 0.5rem;
                font-style: italic;
                color: #a1a1aa;
                font-size: 0.9em;
                margin-top: 0.75rem;
                border: 1px solid rgba(255,255,255,0.06);
            }}
            .reason {{ margin: 0; font-weight: 600; color: #a78bfa; }}
            .time {{ color: #fafafa; font-weight: 600; }}
            a {{ color: #a78bfa; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .total {{
                display: inline-block;
                background: rgba(139,92,246,0.1);
                color: #a78bfa;
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                font-weight: 600;
                margin-bottom: 1rem;
            }}
        </style>
    </head>
    <body>
        <div style="margin-bottom: 2rem;">
            <a href="/" style="font-weight: 700; font-size: 1.25rem; color: #fafafa; text-decoration: none; display: flex; align-items: center; gap: 0.5rem;">
                 <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #a78bfa;"><path d="M4 11v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"></path><path d="M4 11l8-8 8 8"></path><path d="M12 19v-6"></path></svg>
                 Back to Dashboard
            </a>
        </div>
        <h1>Ad Report</h1>
        <h2>{html.escape(str(ep.title))}</h2>
        <p class="meta">GUID: {html.escape(str(ep.guid))}</p>

        <h3>{'Final removal intervals' if analysis else 'Detected Segments'}</h3>
        <p class="total">Total Segments: {len(ad_segments)}</p>

        {rows_html}

        {_timeline_sections(analysis, edit_policy) if analysis else ''}

        <h3>Transcript</h3>
        <p><a href="/artifacts/transcript/{ep.id}" class="btn">View Full Transcript (JSON)</a></p>
    </body>
    </html>
    """

    return html_content
