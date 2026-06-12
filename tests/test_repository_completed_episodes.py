from app.infra.database import get_db_connection, init_db
from app.infra.repository import EpisodeRepository


def seed_completed_episode_rows():
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO subscriptions (id, feed_url, title, slug, image_url)
            VALUES
                (1, 'https://example.com/one.xml', 'Show One', 'show-one', 'https://example.com/one.jpg'),
                (2, 'https://example.com/two.xml', 'Show Two', 'show-two', 'https://example.com/two.jpg')
            """
        )
        conn.execute(
            """
            INSERT INTO episodes (
                subscription_id, guid, title, pub_date, original_url, duration,
                status, local_filename, file_size
            )
            VALUES
                (1, 'old', 'Old Complete', '2026-01-01T10:00:00', 'https://cdn.example.com/old.mp3', 60, 'completed', '/tmp/old.mp3', 10),
                (1, 'new', 'New Complete', '2026-01-02T10:00:00', 'https://cdn.example.com/new.mp3', 60, 'completed', '/tmp/new.mp3', 10),
                (1, 'pending', 'Pending Episode', '2026-01-03T10:00:00', 'https://cdn.example.com/pending.mp3', 60, 'pending', NULL, NULL),
                (2, 'other', 'Other Show', '2026-01-04T10:00:00', 'https://cdn.example.com/other.mp3', 60, 'completed', '/tmp/other.mp3', 10)
            """
        )
        conn.commit()


def test_get_completed_by_subscription_filters_and_orders(isolated_data_dir):
    init_db()
    seed_completed_episode_rows()

    rows = EpisodeRepository().get_completed_by_subscription(1)

    assert [row["guid"] for row in rows] == ["new", "old"]
    assert all(row["subscription_id"] == 1 for row in rows)
    assert all(row["status"] == "completed" for row in rows)


def test_get_completed_with_subscription_info_includes_feed_metadata(isolated_data_dir):
    init_db()
    seed_completed_episode_rows()

    rows = EpisodeRepository().get_completed_with_subscription_info()

    assert [row["guid"] for row in rows] == ["other", "new", "old"]
    assert rows[0]["podcast_title"] == "Show Two"
    assert rows[0]["podcast_slug"] == "show-two"
    assert rows[0]["podcast_image"] == "https://example.com/two.jpg"
