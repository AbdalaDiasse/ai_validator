import base64
import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient
import src.app.main as main_module
from src.app.main import app

# Dummy validator that always returns correct label and high confidence
class DummyValidator:
    def validate(self, image, class_name):
        return {'label': class_name, 'confidence': 0.9}

@pytest.fixture(autouse=True)
def patch_validators(monkeypatch):
    # Replace all real validators with DummyValidator
    monkeypatch.setitem(main_module.VALIDATORS, 'openai', DummyValidator())
    monkeypatch.setitem(main_module.VALIDATORS, 'gemini', DummyValidator())
    monkeypatch.setitem(main_module.VALIDATORS, 'nvidia', DummyValidator())

client = TestClient(app)

def get_sample_image_base64():
    # Create a simple 10x10 red image
    img = Image.new('RGB', (10, 10), color='red')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def test_validate_success():
    payload = {
        'image_base64': get_sample_image_base64(),
        'bbox': {'x': 0, 'y': 0, 'width': 10, 'height': 10},
        'class_name': 'license_plate',
        'validators': ['openai', 'gemini', 'nvidia']
    }
    response = client.post('/validate', json=payload)
    assert response.status_code == 200
    data = response.json()
    for key in payload['validators']:
        assert key in data
        assert data[key]['label'] == 'license_plate'
        assert data[key]['confidence'] >= 1 and data[key]['confidence'] <= 10
        assert data[key]['validated'] is True


def test_unknown_validator():
    payload = {
        'image_base64': get_sample_image_base64(),
        'bbox': {'x': 0, 'y': 0, 'width': 10, 'height': 10},
        'class_name': 'age',
        'validators': ['foo']
    }
    response = client.post('/validate', json=payload)
    assert response.status_code == 400
    assert 'Unknown validator' in response.json()['detail']


def test_invalid_image():
    payload = {
        'image_base64': 'not-a-valid-base64',
        'bbox': {'x': 0, 'y': 0, 'width': 10, 'height': 10},
        'class_name': 'gender'
    }
    response = client.post('/validate', json=payload)
    assert response.status_code == 400