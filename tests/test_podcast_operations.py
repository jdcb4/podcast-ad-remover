import pytest
from app.core import podcast_operations as ops
from app.infra.database import init_db, get_db_connection
from app.infra.repository import JobRepository, EpisodeRepository


@pytest.fixture
def show(isolated_data_dir, monkeypatch):
    init_db()
    with get_db_connection() as conn:
        sid = conn.execute("INSERT INTO subscriptions(feed_url,title,slug,is_active) VALUES('https://example.org/feed','Show','show',1)").lastrowid
        conn.commit()
    entries = [dict(guid=str(i),title=f'Episode {i}',pub_date=f'2020-01-0{i}',original_url=f'https://example.org/{i}.mp3',duration=60,description='',file_size=0) for i in (3,1,2)]
    monkeypatch.setattr(ops, 'discover_history', lambda url: entries)
    return sid, entries


def test_archive_order_pause_cancel_keep_retention(show):
    sid, entries = show
    p = ops.preview(sid, 'archive')
    batch = ops.confirm(sid, p['preview_id'])['batch_id']
    with get_db_connection() as conn:
        assert conn.execute('SELECT keep_whole_show,inherit_retention FROM subscriptions WHERE id=?',(sid,)).fetchone()[:] == (1,0)
    ops.control(sid,batch,'pause')
    assert JobRepository().claim_due(1) == []
    ops.control(sid,batch,'resume')
    claimed = JobRepository().claim_due(1)
    assert claimed[0]['guid'] == '1'
    ops.control(sid,batch,'cancel')
    assert JobRepository().claim_due(5) == []
    with get_db_connection() as conn:
        assert conn.execute("SELECT count(*) FROM jobs WHERE status='running'").fetchone()[0] == 1
        assert conn.execute('SELECT keep_whole_show FROM subscriptions').fetchone()[0] == 1


def test_feed_review_alias_and_preserved_identity(show):
    sid, entries = show
    with get_db_connection() as conn:
        eid = conn.execute("INSERT INTO episodes(subscription_id,guid,title,original_url,status,local_filename) VALUES(?,'old-guid','Old','https://old.org/audio','completed','preserve.mp3')", (sid,)).lastrowid
        conn.commit()
    p = ops.preview(sid,'feed','https://new.org/feed')
    with pytest.raises(ValueError, match='Review every'):
        ops.confirm(sid,p['preview_id'])
    ops.confirm(sid,p['preview_id'],{'1':eid,'2':None,'3':None})
    with get_db_connection() as conn:
        assert conn.execute('SELECT guid,local_filename FROM episodes WHERE id=?',(eid,)).fetchone()[:] == ('old-guid','preserve.mp3')
        assert conn.execute('SELECT count(*) FROM jobs').fetchone()[0] == 0
    assert not EpisodeRepository().create_or_ignore(dict(entries[1],subscription_id=sid,status='pending'))


def test_stale_preview_and_duplicate_destinations(show):
    sid, entries = show
    p = ops.preview(sid,'archive')
    with get_db_connection() as conn:
        conn.execute("UPDATE subscriptions SET title='Changed' WHERE id=?",(sid,))
        conn.commit()
    with pytest.raises(ValueError, match='changed'):
        ops.confirm(sid,p['preview_id'])
    with get_db_connection() as conn:
        conn.execute("INSERT INTO subscriptions(feed_url,title,slug) VALUES('https://new.org/feed','Other','other')")
        conn.commit()
    with pytest.raises(ValueError, match='already uses'):
        ops.preview(sid,'feed','https://new.org/feed')


def test_no_double_mapping_and_atomic_rollback(show):
    sid, entries = show
    with get_db_connection() as conn:
        eid=conn.execute("INSERT INTO episodes(subscription_id,guid,title,original_url,status) VALUES(?,'old','Old','https://old.org/audio','completed')",(sid,)).lastrowid
        conn.commit()
    p=ops.preview(sid,'feed','https://new.org/feed')
    with pytest.raises(ValueError, match='Two incoming'):
        ops.confirm(sid,p['preview_id'],{'1':eid,'2':eid,'3':None})
    with get_db_connection() as conn:
        assert conn.execute('SELECT count(*) FROM episode_source_aliases').fetchone()[0] == 0
        assert conn.execute('SELECT feed_url FROM subscriptions').fetchone()[0] == 'https://example.org/feed'


def test_archive_preserves_window_exclusions_and_normal_priority(show):
    sid, entries = show
    with get_db_connection() as conn:
        conn.execute('UPDATE app_settings SET default_retention_limit=5,default_retention_days=60 WHERE id=1')
        conn.execute('UPDATE subscriptions SET inherit_retention=1,retention_limit=1 WHERE id=?',(sid,))
        conn.execute("INSERT INTO episodes(subscription_id,guid,title,status,original_url) VALUES(?,'1','Deleted','ignored','https://example.org/1.mp3')",(sid,))
        conn.execute("INSERT INTO episodes(subscription_id,guid,title,status,original_url) VALUES(?,'2','Done','completed','https://example.org/2.mp3')",(sid,))
        conn.commit()
    p=ops.preview(sid,'archive')
    assert p['counts']==dict(completed=1,active=0,excluded=1,eligible=1)
    batch=ops.confirm(sid,p['preview_id'])['batch_id']
    with get_db_connection() as conn:
        assert conn.execute('SELECT retention_limit,retention_days FROM subscriptions').fetchone()[:]==(5,60)
    EpisodeRepository().create_or_ignore(dict(entries[0],guid='normal',subscription_id=sid,status='pending'))
    claimed=JobRepository().claim_due(1)[0]
    assert claimed['guid']=='normal'
    with pytest.raises(ValueError,match='existing archive'):
        ops.preview(sid,'archive')
    ops.control(sid,batch,'pause')
    JobRepository().repair_missing_active_jobs()
    assert JobRepository().claim_due(5)==[]


def test_retry_pause_restart_and_cancelled_failure(show):
    from datetime import datetime, timedelta
    sid,_=show
    batch=ops.confirm(sid,ops.preview(sid,'archive')['preview_id'])['batch_id']
    job=JobRepository().claim_due(1)[0]
    EpisodeRepository().update_retry(job['id'],1,datetime.now()+timedelta(hours=1),'Unavailable')
    assert JobRepository().claim_due(1)[0]['guid']=='2'
    ops.control(sid,batch,'pause')
    with get_db_connection() as conn:
        conn.execute("UPDATE jobs SET updated_at='2000-01-01',locked_at='2000-01-01' WHERE status='running'")
        conn.commit()
    JobRepository().recover_stale_running()
    assert JobRepository().claim_due(5)==[]
    ops.control(sid,batch,'resume')
    job=JobRepository().claim_due(1)[0]
    ops.control(sid,batch,'cancel')
    EpisodeRepository().update_retry(job['id'],2,datetime.now(),'Unavailable')
    JobRepository().repair_missing_active_jobs()
    assert JobRepository().claim_due(5)==[]
    with get_db_connection() as conn:
        assert conn.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running','retry_scheduled')").fetchone()[0]==0


def test_automatic_deletion_rechecks_retention(show):
    sid,_=show
    batch=ops.confirm(sid,ops.preview(sid,'archive')['preview_id'])['batch_id']
    job=JobRepository().claim_due(1)[0]
    assert not EpisodeRepository().request_deletion(job['id'],automatic=True)
    assert EpisodeRepository().request_deletion(job['id'])


def test_old_refresh_cannot_queue_after_replacement(show):
    sid,entries=show
    old=dict(entries[0],subscription_id=sid,status='pending',_source_feed_url='https://example.org/feed')
    preview=ops.preview(sid,'feed','https://new.org/feed')
    ops.confirm(sid,preview['preview_id'],{str(i):None for i in (1,2,3)})
    assert not EpisodeRepository().create_or_ignore(old)
    EpisodeRepository().update_status_by_guid(sid,'3','pending','unprocessed',expected_feed_url=old['_source_feed_url'])
    with get_db_connection() as conn:
        assert conn.execute('SELECT count(*) FROM jobs').fetchone()[0]==0


def test_destination_race_and_expired_preview_are_atomic(show):
    sid,_=show
    p=ops.preview(sid,'feed','https://new.org/feed')
    with get_db_connection() as conn:
        conn.execute("INSERT INTO subscriptions(feed_url,title,slug) VALUES('https://new.org/feed','Other','other')")
        conn.commit()
    with pytest.raises(ValueError,match='already uses'):
        ops.confirm(sid,p['preview_id'],{str(i):None for i in (1,2,3)})
    with get_db_connection() as conn:
        assert conn.execute('SELECT count(*) FROM episodes').fetchone()[0]==0
        conn.execute('UPDATE podcast_previews SET expires_at=0')
        conn.commit()
    with pytest.raises(ValueError,match='expired'):
        ops.confirm(sid,p['preview_id'])


def test_api_permissions_and_batch_roundtrip(show):
    from tests.test_ai_api import create_user
    from tests.conftest import make_client,enable_ai_api,auth_header
    from app.infra.repository import ApiTokenRepository
    sid,_=show
    enable_ai_api()
    owner=create_user('owner');other=create_user('other')
    with get_db_connection() as conn:
        conn.execute('UPDATE subscriptions SET owner_user_id=? WHERE id=?',(owner,sid));conn.commit()
    client=make_client();path=f'/api/v1/subscriptions/{sid}'
    for user,scope,code in [(other,'write',403),(owner,'read',403),(owner,'write',200)]:
        token=ApiTokenRepository().create('test',scopes=[scope],user_id=user)
        result=client.post(path+'/operations/preview',json={'kind':'archive'},headers=auth_header(token))
        assert result.status_code==code
    headers=auth_header(token)
    confirmed=client.post(path+'/operations/confirm',json={'preview_id':result.json()['preview_id']},headers=headers)
    assert confirmed.status_code==200
    bid=confirmed.json()['batch_id']
    assert client.post(path+f'/archive-batches/{bid}',json={'action':'pause'},headers=headers).status_code==200
    assert client.get(path+'/archive-batches',headers=headers).json()[0]['status']=='paused'


def test_history_pagination_and_loop(monkeypatch):
    def feed(guid,next_url=''):
        link=f'<atom:link rel="next" href="{next_url}"/>' if next_url else ''
        return f'<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom"><channel><title>Show</title>{link}<item><title>{guid}</title><guid>{guid}</guid><enclosure url="https://example.org/{guid}.mp3" type="audio/mpeg" length="1"/></item></channel></rss>'.encode()
    monkeypatch.setattr(ops.FeedManager,'_fetch_feed',lambda url:feed('1','/two') if url.endswith('/one') else feed('2'))
    assert [x['guid'] for x in ops.discover_history('https://example.org/one')]==['1','2']
    monkeypatch.setattr(ops.FeedManager,'_fetch_feed',lambda url:feed('1','/one'))
    with pytest.raises(ValueError,match='loops'):
        ops.discover_history('https://example.org/one')


def test_missing_dates_last_and_repair_cannot_escape_pause(show):
    sid,entries=show
    entries[1]['pub_date']=None
    batch=ops.confirm(sid,ops.preview(sid,'archive')['preview_id'])['batch_id']
    with get_db_connection() as conn:
        order=conn.execute('SELECT e.guid FROM jobs j JOIN episodes e ON e.id=j.episode_id ORDER BY j.archive_position').fetchall()
        assert [r[0] for r in order]==['2','3','1']
        conn.execute("UPDATE jobs SET status='cancelled' WHERE episode_id=(SELECT id FROM episodes WHERE guid='2')")
        conn.commit()
    ops.control(sid,batch,'pause')
    JobRepository().repair_missing_active_jobs()
    assert JobRepository().claim_due(5)==[]
    with get_db_connection() as conn:
        assert conn.execute('SELECT count(*) FROM jobs').fetchone()[0]==3
        assert conn.execute('SELECT count(*) FROM jobs WHERE archive_batch_id=?',(batch,)).fetchone()[0]==3


def test_exact_guid_and_unique_enclosure_match_without_review(show):
    sid,entries=show
    with get_db_connection() as conn:
        first=conn.execute("INSERT INTO episodes(subscription_id,guid,title,original_url,status) VALUES(?,'1','One','https://old.org/one','completed')",(sid,)).lastrowid
        second=conn.execute("INSERT INTO episodes(subscription_id,guid,title,original_url,status) VALUES(?,'old-two','Two','https://example.org/2.mp3','completed')",(sid,)).lastrowid
        conn.commit()
    p=ops.preview(sid,'feed','https://new.org/feed')
    matches={x['entry']['guid']:x['episode_id'] for x in p['items']}
    assert matches=={'1':first,'2':second,'3':None}
    ops.confirm(sid,p['preview_id'],{'3':None})
    with get_db_connection() as conn:
        assert conn.execute('SELECT guid FROM episodes WHERE id=?',(second,)).fetchone()[0]=='old-two'
        assert conn.execute('SELECT count(*) FROM jobs').fetchone()[0]==0
