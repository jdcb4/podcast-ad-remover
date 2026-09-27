from pathlib import Path


def test_sponsorblock_execution_removed():
    source = Path('app/core/processor.py').read_text(encoding='utf-8')
    assert 'sponsorblock' not in source.lower()
    assert not Path('app/core/sponsorblock.py').exists()
