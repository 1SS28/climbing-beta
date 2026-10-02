"""Measurement API regressions. Run: .venv/bin/python -m unittest test_measurement"""
import io
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import server


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)
        server._cache['test'] = {'size': (1000, 1000), 'polys': []}
        self.reference = {'method': 'reference', 'points': [[0, 0], [1000, 0]], 'metres': 1}
        self.body = {'id': 'test', 'points': [[100, 800], [400, 400]], 'calibration': self.reference}

    def tearDown(self):
        server._cache.pop('test', None)

    def post(self, **changes):
        return self.client.post('/api/measure', json=dict(self.body, **changes))

    def test_manual_points_without_detections(self):
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['gaps'][0]['metres'], .5)
        self.assertEqual(response.json()['rise_m'], .4)
        self.assertIsNone(response.json()['warning'])  # A cropped route isn't an invalid scale.

    def test_grid_removes_perspective(self):
        H = np.array([[300, 50, 100], [20, -300, 800], [.15, .05, 1]])
        def project(points):
            q = np.c_[points, np.ones(len(points))] @ H.T
            return (q[:, :2] / q[:, 2:]).tolist()
        grid = {'method': 'grid', 'corners': project([[0, 2], [2, 2], [0, 0], [2, 0]]),
                'cols': 10, 'rows': 10, 'spacing_m': .2}
        response = self.post(points=project([[.2, .2], [.8, 1]]), calibration=grid)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['gaps'][0]['metres'], 1)

    def test_no_implicit_scale(self):
        for value in [None, {}, {'method': 'reference', 'points': [[0, 0]], 'metres': 1}]:
            self.assertEqual(self.post(calibration=value).status_code, 422)
        body = dict(self.body); body.pop('calibration')
        self.assertEqual(self.client.post('/api/measure', json=body).status_code, 422)

    def test_invalid_reference(self):
        for metres in [0, -1]:
            self.assertEqual(self.post(calibration=dict(self.reference, metres=metres)).status_code, 422)
        self.assertEqual(self.post(calibration=dict(self.reference, points=[[2, 2], [2, 2]])).status_code, 400)
        self.assertEqual(self.post(points=[[0, 0], [1001, 0]]).status_code, 400)
        self.assertEqual(self.post(points=[[0, 0]]).status_code, 422)
        self.assertEqual(self.post(points=[[-1, 0], [0, 0]]).status_code, 422)
        self.assertEqual(self.post(points=[[0, 0], ['NaN', 2]]).status_code, 422)

    def test_degenerate_grid(self):
        for corners in [[[0, 0]] * 4, [[0, 0], [100, 100], [100, 0], [0, 100]],
                        [[0, 0], [1, 0], [2, 0], [3, 0]]]:
            grid = {'method': 'grid', 'corners': corners, 'cols': 2, 'rows': 2, 'spacing_m': .2}
            self.assertEqual(self.post(calibration=grid).status_code, 400)

    def test_hold_ids_are_validated(self):
        for route, status in [([-1, 0], 422), ([0, 1], 400), ([1.5, 2], 422)]:
            self.assertEqual(self.post(points=None, route=route).status_code, status)
        server._cache['test']['polys'] = [np.array([[100, 100], [100, 200], [200, 200], [200, 100]]),
                                        np.array([[400, 500], [400, 600], [500, 600], [500, 500]])]
        self.assertEqual(self.post(points=None, route=[0, 1]).json()['gaps'][0]['metres'], .5)
        self.assertEqual(self.post(route=[0, 1]).status_code, 422)

    def test_unknown_photo(self):
        self.assertEqual(self.post(id='missing').status_code, 404)

    def test_bad_photo(self):
        response = self.client.post('/api/photo', files={'file': ('broken.jpg', b'bad', 'image/jpeg')})
        self.assertEqual(response.status_code, 400)

    def test_upload_corrects_phone_orientation_and_handles_no_holds(self):
        raw = io.BytesIO(); img = Image.new('RGB', (20, 40)); exif = Image.Exif(); exif[274] = 6
        img.save(raw, format='JPEG', exif=exif)
        fake_model = SimpleNamespace(predict=lambda *a, **kw: [SimpleNamespace(masks=None)])
        with patch.object(server, 'model', return_value=fake_model):
            response = self.client.post('/api/photo', files={'file': ('phone.jpg', raw.getvalue(), 'image/jpeg')})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        try:
            self.assertEqual((result['width'], result['height']), (40, 20))
            self.assertEqual(result['holds'], [])
        finally:
            server._cache.pop(result['id'], None)
            (server.UPLOADS / f"{result['id']}.jpg").unlink()


if __name__ == '__main__':
    unittest.main()
