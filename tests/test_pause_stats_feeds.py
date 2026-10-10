import asyncio
from pathlib import Path
from xml.etree.ElementTree import fromstring

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from app.core.models import User
from app.core.processor import Processor
from app.core.rss_gen import RSSGenerator
from app.core.statistics import get_statistics, record_completion
from app.infra.database import get_db_connection, init_db
from app.infra.repository import EpisodeRepository, JobRepository
from app.web import router as web
from app.web.auth import require_auth, require_admin


@pytest.fixture
def sample(isolated_data_dir):
    init_db()
    with get_db_connection() as conn:
        users = [conn.execute("INSERT INTO users(username,password_hash,is_admin) VALUES(?,?,?)", (name,'test',admin)).lastrowid for name,admin in [('joe',1),('listener',0),('other',0)]]
        shows = [conn.execute("INSERT INTO subscriptions(feed_url,title,slug) VALUES(?,?,?)", (f'https://example.org/{name}',name,name)).lastrowid for name in ['one','two']]
        conn.execute("INSERT INTO user_subscriptions(user_id,subscription_id,added_at) VALUES(?,?,'2000-01-01')", (users[1],shows[0]))
        episodes = [conn.execute("INSERT INTO episodes(subscription_id,guid,title,original_url,status,duration,output_duration,local_filename,processed_at) VALUES(?,?,?,'https://example.org/audio','completed',3600,3000,?,CURRENT_TIMESTAMP)", (sid,str(sid),name,str(Path(isolated_data_dir)/'podcasts'/name/'audio.mp3'))).lastrowid for sid,name in zip(shows,['one','two'])]
        for eid in episodes:
            record_completion(conn,eid,12)
        conn.commit()
    app = FastAPI()
    app.add_middleware(SessionMiddleware,secret_key='test')
    app.include_router(web.router)
    return TestClient(app),app,users,shows,episodes


def test_pause_blocks_claims_and_cleanup_but_manual_delete_works(sample):
    client,app,users,shows,episodes = sample
    repo=EpisodeRepository()
    with get_db_connection() as conn:
        conn.execute("UPDATE episodes SET status='processing' WHERE id=?", (episodes[0],))
        conn.commit()
    repo.update_status(episodes[0],'pending')
    response=client.post(f'/subscriptions/{shows[0]}/pause',data={'processing_paused':'true'},follow_redirects=False)
    assert response.status_code==303
    assert JobRepository().claim_due(5)==[]
    assert not repo.request_deletion(episodes[0],automatic=True)
    client.post(f'/subscriptions/{shows[0]}/pause',data={'processing_paused':'false'})
    claimed=JobRepository().claim_due(1)
    assert claimed[0]['id']==episodes[0]
    client.post('/admin/system/update',data={'section':'system','processing_paused':'true'},follow_redirects=False)
    assert web.get_global_settings()['processing_paused']==1
    assert not repo.request_deletion(episodes[1],automatic=True)
    assert repo.request_deletion(episodes[1])
    # Saving another settings section must preserve the pause.
    client.post('/admin/system/update',data={'section':'api'})
    assert web.get_global_settings()['processing_paused']==1
    page=client.get('/?view=library').text
    assert ' (processing paused)' in page
    assert 'Processing paused' in page


def test_pause_discovery_and_bulk_permissions(sample,monkeypatch):
    client,app,users,shows,episodes=sample
    monkeypatch.setattr(Processor,'__init__',lambda self: None)
    processor=Processor()
    processor.sub_repo=web.sub_repo
    with get_db_connection() as conn:
        conn.execute('UPDATE app_settings SET processing_paused=1')
        conn.commit()
    # No adapter/episode repository exists: paused discovery must return first.
    asyncio.run(processor.check_feeds())
    regular=User(id=users[1],username='listener',password_hash='',is_admin=False)
    app.dependency_overrides[require_auth]=lambda: regular
    result=client.post('/subscriptions/bulk-settings',data={'subscription_ids':shows,'pause_mode':'pause'},follow_redirects=False)
    assert result.status_code==403
    # A real regular session also fails the single-feed admin dependency.
    with get_db_connection() as conn:
        conn.execute('UPDATE app_settings SET auth_enabled=1')
        conn.commit()
    assert client.post(f'/subscriptions/{shows[0]}/pause',data={'processing_paused':True}).status_code==401


def test_personal_feed_membership_and_auth(sample):
    client,app,users,shows,episodes=sample
    global_xml=fromstring(Path(RSSGenerator().generate_unified_feed()).read_text())
    assert len(global_xml.findall('channel/item'))==2
    url=f'/feed/users/{users[1]}/unified.xml'
    assert len(fromstring(client.get(url).text).findall('channel/item'))==1
    with get_db_connection() as conn:
        conn.execute('DELETE FROM user_subscriptions WHERE user_id=?',(users[1],))
        conn.execute('UPDATE app_settings SET enable_feed_auth=1')
        conn.commit()
    token=web.feed_token_repo.create(users[1])
    other=web.feed_token_repo.create(users[2])
    assert client.get(url+'?token='+other).status_code==401
    assert client.get(url).status_code==401
    response=client.get(url+'?token='+token)
    assert response.status_code==200
    assert len(fromstring(response.text).findall('channel/item'))==0
    assert client.get('/feed/unified.xml?token='+other).status_code==200
    web.feed_token_repo.revoke(token)
    assert client.get(url+'?token='+token).status_code==401


def test_personal_stats_survive_cleanup_and_no_reprocessing_double_count(sample):
    client,app,users,shows,episodes=sample
    stats=get_statistics(user_id=users[1],historical=True,period='week')
    assert stats['episodes']==1 and stats['saved']==600
    with get_db_connection() as conn:
        record_completion(conn,episodes[0],20)
        conn.execute('DELETE FROM user_subscriptions WHERE user_id=?',(users[1],))
        conn.execute('DELETE FROM episodes WHERE id=?',(episodes[0],))
        conn.commit()
    assert get_statistics(user_id=users[1])['episodes']==0
    assert get_statistics(user_id=users[1],historical=True)['episodes']==1
    assert get_statistics(user_id=users[2],historical=True)['episodes']==0
    assert get_statistics(historical=True)['episodes']==2
    regular=User(id=users[1],username='listener',password_hash='',is_admin=False)
    app.dependency_overrides[require_auth]=lambda: regular
    page=client.get('/stats').text
    assert 'My Pods' in page
    assert 'Library · Current holdings' not in page
    assert 'LLM / speech calls' not in page
    assert '>Tasks</a>' not in page and '>Settings</a>' not in page


def test_account_feed_uses_selected_collection(sample):
    client,app,users,shows,episodes=sample
    regular=User(id=users[1],username='listener',password_hash='',is_admin=False)
    app.dependency_overrides[require_auth]=lambda: regular
    assert f'/feed/users/{users[1]}/unified.xml' in client.get('/account/feed?view=mine').json()['rss']
    assert '/feed/unified.xml' in client.get('/account/feed?view=library').json()['rss']


def test_completion_is_recorded_while_running_episode_finishes_during_pause(sample):
    client,app,users,shows,episodes=sample
    with get_db_connection() as conn:
        eid=conn.execute("INSERT INTO episodes(subscription_id,guid,title,original_url,status,duration,output_duration) VALUES(?,'new','New','https://example.org/new','pending',120,90)",(shows[0],)).lastrowid
        conn.commit()
    repo=EpisodeRepository()
    repo.update_status(eid,'pending')
    claim=JobRepository().claim_due(1)[0]
    client.post(f'/subscriptions/{shows[0]}/pause',data={'processing_paused':True},follow_redirects=False)
    attempt=EpisodeRepository(attempt=(claim['job_id'],claim['claim_token']))
    attempt.transcription_seconds=5
    attempt.update_status(eid,'completed',filename='new.mp3')
    assert repo.get_by_id(eid).status=='completed'
    totals=get_statistics(user_id=users[1],historical=True)
    assert totals['episodes']==2 and totals['saved']==630
    assert not repo.request_deletion(eid,automatic=True)


def test_migration_import_backup_and_idempotency(sample,isolated_data_dir):
    client,app,users,shows,episodes=sample
    with get_db_connection() as conn:
        conn.execute('DROP TABLE processing_history_users')
        conn.execute('DROP TABLE processing_history')
        conn.execute('ALTER TABLE subscriptions DROP COLUMN processing_paused')
        conn.execute('ALTER TABLE app_settings DROP COLUMN processing_paused')
        conn.execute("DELETE FROM schema_migrations WHERE version='20261010_0027_pause_statistics'")
        conn.commit()
    init_db()
    assert get_statistics(historical=True)['episodes']==2
    assert get_statistics(user_id=users[1],historical=True)['episodes']==1
    assert web.get_global_settings()['processing_paused']==0
    backups=list((isolated_data_dir/'backups').glob('podcasts-before-migration-*.db'))
    assert backups
    init_db()
    assert get_statistics(historical=True)['episodes']==2
    with get_db_connection() as conn:
        assert conn.execute('SELECT SUM(imported) FROM processing_history').fetchone()[0]==2
        assert conn.execute('SELECT COUNT(*) FROM episodes WHERE local_filename IS NOT NULL').fetchone()[0]==2


def test_history_periods_and_usage_coverage(sample):
    client,app,users,shows,episodes=sample
    with get_db_connection() as conn:
        conn.execute("UPDATE processing_history SET processed_at='2000-01-01' WHERE episode_id=?",(episodes[1],))
        conn.execute("INSERT INTO provider_calls(job_id,provider,model,input_tokens,output_tokens,outcome) VALUES(1,'fixture','model',12,4,'success')")
        conn.execute("INSERT INTO provider_calls(job_id,provider,model,outcome) VALUES(1,'fixture','model','failed')")
        conn.commit()
    for period in ['week','month','year']:
        stats=get_statistics(historical=True,period=period)
        assert stats['episodes']==1
        assert stats['usage']['calls']==2
        assert stats['usage']['input_tokens']==12
        assert stats['usage']['known_input']==1
    assert get_statistics(historical=True)['episodes']==2


def test_personal_basic_auth_and_regular_user_admin_controls(sample):
    import base64
    from app.web.auth_utils import hash_password
    client,app,users,shows,episodes=sample
    with get_db_connection() as conn:
        conn.execute("UPDATE users SET password_hash=? WHERE id=?",(hash_password('synthetic-password'),users[1]))
        conn.execute('UPDATE app_settings SET auth_enabled=1,enable_feed_auth=1')
        conn.commit()
    credentials=base64.b64encode(b'listener:synthetic-password').decode()
    personal=f'/feed/users/{users[1]}/unified.xml'
    response=client.get(personal,headers={'Authorization':'Basic '+credentials})
    assert response.status_code==200
    assert len(fromstring(response.text).findall('channel/item'))==1
    assert 'auth=' in fromstring(response.text).find('channel/item/enclosure').get('url')
    assert client.get(f'/feed/users/{users[0]}/unified.xml',headers={'Authorization':'Basic '+credentials}).status_code==401
    assert client.get(personal+'?auth='+credentials).status_code==200
    # Admin page protection belongs to the production auth middleware.
    from app.web.auth import auth_middleware
    protected=FastAPI()
    protected.middleware("http")(auth_middleware)
    protected.add_middleware(SessionMiddleware,secret_key='test')
    protected.include_router(web.router)
    client=TestClient(protected)
    login=client.post('/login',data={'username':'listener','password':'synthetic-password'},follow_redirects=False)

    assert login.status_code in (302,303)
    assert client.post(f'/subscriptions/{shows[0]}/pause',data={'processing_paused':'true'}).status_code==403
    assert client.get('/admin/system').status_code==403
    assert client.get('/admin/queue').status_code==403
    page=client.get('/?view=mine').text
    assert 'My Pods' in page
    assert '>Tasks</a>' not in page and '>Settings</a>' not in page
    page=client.get('/stats').text
    assert 'Library · Current holdings' not in page
    assert 'LLM / speech calls' not in page
