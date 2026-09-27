from pathlib import Path
import re
from zipfile import ZipFile
from app.infra.database import init_db
from app.api.v1.router import router
from app.api.v1.schemas import SubscriptionSettingsUpdate
from scripts.package_agent_skill import build
from tests.conftest import make_client, enable_ai_api

ROOT = Path(__file__).resolve().parents[1]


def normalized(path):
    return re.sub(r'\{[^}]+\}', '{id}', path)


def test_api_guide_covers_actual_endpoints_and_settings():
    guide = (ROOT / 'Documentation/API.md').read_text(encoding='utf-8')
    documented = {(method, normalized(path)) for method, path in re.findall(r'\| `(GET|POST|PATCH|DELETE|PUT)` \| `(/api/v1/[^`]+)`', guide)}
    actual = {(method, normalized('/api/v1' + route.path)) for route in router.routes for method in route.methods}
    assert documented == actual
    for field in SubscriptionSettingsUpdate.model_fields:
        assert f'`{field}`' in guide


def test_openapi_exposes_runtime_scopes(isolated_data_dir):
    init_db()
    enable_ai_api()
    response = make_client().get('/api/v1/openapi.json')
    assert response.status_code == 200
    schema = response.json()
    for route in router.routes:
        scopes = sorted({s for dependency in route.dependant.dependencies for s in getattr(dependency.call, 'required_scopes', [])})
        for method in route.methods:
            operation = schema['paths']['/api/v1' + route.path][method.lower()]
            assert operation['x-required-scopes'] == scopes
            if scopes:
                assert operation['security'] == [{'HTTPBearer': []}]
                assert 'Retry-After' in operation['responses']['429']['headers']
    assert schema['paths']['/api/v1/subscriptions/import']['post']['x-required-scopes'] == ['write']
    assert schema['components']['schemas']['PodcastImportRequest']['properties']['dry_run']['default'] is True


def test_agent_bundle_embeds_current_reference(tmp_path):
    path = build(tmp_path / 'skill.zip')
    with ZipFile(path) as archive:
        names = archive.namelist()
        assert 'podcast-ad-remover/SKILL.md' in names
        assert 'podcast-ad-remover/agents/openai.yaml' in names
        assert 'podcast-ad-remover/references/operations.md' in names
        assert archive.read('podcast-ad-remover/references/API.md') == (ROOT / 'Documentation/API.md').read_bytes()
        assert all(not name.startswith('/') and '..' not in Path(name).parts for name in names)
        assert all(name.endswith(('.md', '.yaml')) for name in names)
