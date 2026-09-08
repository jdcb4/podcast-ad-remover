"""One durable request budget shared by analysis, summaries and remote speech."""
import time
from contextlib import contextmanager
from contextvars import ContextVar

from app.core.config import settings
from app.infra.database import get_db_connection


class ProviderBudgetExceeded(RuntimeError):
    pass


current_claim = ContextVar('provider_budget_claim', default=None)


@contextmanager
def provider_request(provider: str, model: str):
    claim = current_claim.get()
    call_id = None
    started = time.monotonic()
    if claim:
        job_id, token = claim
        with get_db_connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            updated = conn.execute("""UPDATE jobs SET provider_call_count=provider_call_count+1
                WHERE id=? AND locked_by=? AND status='running' AND cancel_requested=0
                AND provider_call_count < ?""", (job_id, token, settings.MAX_PROVIDER_CALLS_PER_JOB))
            if not updated.rowcount:
                raise ProviderBudgetExceeded('Provider request budget exhausted or processing cancelled; review the job before retrying')
            call_id = conn.execute('INSERT INTO provider_calls(job_id,provider,model) VALUES(?,?,?)', (job_id, provider, model)).lastrowid
            conn.commit()
    metrics = {'input_tokens': None, 'output_tokens': None}
    outcome = 'failed'
    try:
        yield metrics
        outcome = 'succeeded'
    finally:
        if call_id:
            with get_db_connection() as conn:
                conn.execute('UPDATE provider_calls SET duration_ms=?, outcome=?, input_tokens=?, output_tokens=? WHERE id=?',
                             (round((time.monotonic() - started) * 1000), outcome, metrics['input_tokens'], metrics['output_tokens'], call_id))
                conn.commit()
