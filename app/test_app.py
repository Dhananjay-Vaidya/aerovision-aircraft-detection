"""No weights or GPU are loaded by these tests."""
import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import server


@pytest.fixture
def client():
    with TestClient(server.app) as client:
        yield client


@pytest.mark.parametrize('name,conf,expected,reason', [
    ('shared', .99, 'uncertain', 'shared airframe'),
    ('friendly', .49, 'uncertain', 'low confidence'),
    ('friendly', .80, 'allied', None),
    ('hostile', .80, 'adversary', None),
    ('friendly', .50, 'allied', None),
    ('other_aircraft', .99, 'uncertain', 'other aircraft'),
    ('unmapped', .99, 'uncertain', 'other aircraft'),
])
def test_decision(name, conf, expected, reason):
    _, decision, actual_reason = server.decide(name, conf,
        {'shared': 'ambiguous', 'friendly': 'allied', 'hostile': 'adversary', 'other_aircraft': 'unknown'}, .5)
    assert (decision, actual_reason) == (expected, reason)


def test_models_without_weights(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, 'MODELS_DIR', tmp_path)
    response = client.get('/api/models')
    assert response.status_code == 200
    assert response.json()['models'] == []
    assert response.json()['default'] is None
    assert len(response.json()['corruptions']) == 9


def test_models_and_thresholds(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, 'MODELS_DIR', tmp_path)
    (tmp_path / 'yolo26s_degrade.pt').touch()
    result = client.get('/api/models').json()
    assert result['default'] == 'yolo26s_degrade'
    expected = json.loads((server.ROOT / 'paper_assets/numbers.json').read_text())
    assert result['models'][0]['tau'] == expected['abstention_tau']['yolo26s_degrade']['matched']
    assert server.settings(None, .10)[1] == result['models'][0]['tau']['0.10']


def test_unreadable_image(client):
    response = client.post('/api/detect', files={'file': ('bad.jpg', b'not an image')})
    assert response.status_code == 422
    assert 'Unreadable' in response.json()['detail']


def fake_detect(img, model=None, risk=.10):
    return {'image_size': {'width': img.shape[1], 'height': img.shape[0]}, 'inference_ms': 1.,
            'detections': [], 'counts': dict(allied=0, adversary=0, uncertain=0)}


@pytest.mark.parametrize('corruption', ['none', *server.CORRUPTIONS])
def test_image_pixels_and_fixed_seed(client, monkeypatch, corruption):
    seen = []
    def record(img, model, risk):
        seen.append(img.copy())
        return fake_detect(img)
    monkeypatch.setattr(server, 'detect', record)
    img = np.random.default_rng(5).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    content = server.encode(img, '.png')
    for _ in range(2):
        response = client.post('/api/detect', files={'file': ('test.png', content)},
                               data={'corruption': corruption, 'severity': 2})
        assert response.status_code == 200
        displayed = server.decode(base64.b64decode(response.json()['image_base64']))
        np.testing.assert_array_equal(displayed, seen[-1])
    np.testing.assert_array_equal(seen[0], seen[1])


def test_missing_model_and_invalid_risk(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, 'MODELS_DIR', tmp_path)
    content = server.encode(np.zeros((20, 20, 3), np.uint8))
    assert client.post('/api/detect', files={'file': ('a.jpg', content)}).status_code == 404
    (tmp_path / 'yolo26s_degrade.pt').touch()
    assert client.post('/api/detect', files={'file': ('a.jpg', content)}, data={'risk': '.2'}).status_code == 422


def test_bad_camera_url(client, monkeypatch):
    monkeypatch.setattr(server, 'settings', lambda *args: ('test', .5))
    response = client.post('/api/camera/start', json={'source': 'not-a-camera'})
    assert response.status_code == 422
    assert 'camera URL' in response.json()['detail']


def test_camera_failure_status(client, monkeypatch):
    monkeypatch.setattr(server, 'settings', lambda *args: ('test', .5))
    class BadCapture:
        def isOpened(self): return False
        def release(self): pass
    monkeypatch.setattr(server.cv2, 'VideoCapture', lambda *args: BadCapture())
    assert client.post('/api/camera/start', json={'source': 'rtsp://127.0.0.1/absent'}).status_code == 200
    server.camera.thread.join(timeout=2)
    assert 'Cannot open camera' in client.get('/api/camera/status').json()['error']
    assert client.post('/api/camera/stop', json={'source': '0'}).status_code == 200


def test_websocket(client, monkeypatch):
    monkeypatch.setattr(server, 'settings', lambda *args: ('test', .5))
    monkeypatch.setattr(server, 'detect', fake_detect)
    with client.websocket_connect('/ws/stream') as ws:
        ws.send_bytes(server.encode(np.zeros((24, 32, 3), np.uint8)))
        assert ws.receive_json()['image_size'] == {'width': 32, 'height': 24}
        ws.send_bytes(b'bad image')
        assert 'Unreadable' in ws.receive_json()['error']


def test_unreadable_video(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, 'settings', lambda *args: ('test', .5))
    monkeypatch.setattr(server, 'OUTPUTS', tmp_path)
    response = client.post('/api/video', files={'file': ('bad.mp4', b'bad video')})
    assert response.status_code == 422
    assert not list(tmp_path.iterdir())


def test_video_encoding_and_timing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, 'settings', lambda *args: ('test', .5))
    detection = dict(box=[5, 20, 45, 40], **{'class': 'Rafale'}, conf=.4,
                     affiliation='allied', decision='uncertain', reason='low confidence')
    def uncertain_detect(img, model=None, risk=.10):
        return {**fake_detect(img), 'tau': .5, 'risk': risk,
                'detections': [detection], 'counts': dict(allied=0, adversary=0, uncertain=1)}
    monkeypatch.setattr(server, 'detect', uncertain_detect)
    source = tmp_path / 'fixture.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    assert writer.isOpened()
    for index in range(6):
        writer.write(np.full((48, 64, 3), index * 30, np.uint8))
    writer.release()
    out = tmp_path / 'out'
    out.mkdir()
    monkeypatch.setattr(server, 'OUTPUTS', out)
    response = client.post('/api/video', files={'file': ('test.avi', source.read_bytes())}, data={'every_n': 2})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['tau'] == .5
    assert result['risk'] == .10
    assert all(row['detections'] == [detection] for row in result['frames'])
    assert [row['frame'] for row in result['frames']] == [0, 2, 4]
    cap = cv2.VideoCapture(str(out / result['url'].split('/')[-1]))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 3
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(5)
    assert int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little') in (b'avc1', b'h264')
    cap.release()
    assert len(list(out.iterdir())) == 1


def test_static_page(client):
    assert 'AeroVision Live' in client.get('/').text
    assert client.get('/static/app.js').status_code == 200
