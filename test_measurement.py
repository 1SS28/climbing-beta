"""Measurement API regressions. Run: .venv/bin/python -m unittest test_measurement"""
import io
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import measurement
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


def square(cx, cy, half):
    """A square mask, the simplest polygon with an area."""
    return np.array([[cx - half, cy - half], [cx + half, cy - half],
                     [cx + half, cy + half], [cx - half, cy + half]], dtype=float)


# Straight-on 2 x 2 m block at 250 px/m: wall x = (u - 100) / 250, y = (1000 - v) / 250.
FLAT = {'method': 'grid', 'corners': [[100, 500], [600, 500], [100, 1000], [600, 1000]],
        'cols': 10, 'rows': 10, 'spacing_m': .2}
FLAT_SIZE = (700, 1100)

# The same block on a wall 60 degrees off square, which puts the vanishing line
# inside the frame. Rows below v=800 run away; below v=960 they cannot be placed.
TILTED = {'method': 'grid', 'corners': [[233.3, 286.7], [766.7, 286.7],
                                        [330.9, 533.8], [669.1, 533.8]],
          'cols': 10, 'rows': 10, 'spacing_m': .2}
TILTED_SIZE = (1000, 1000)


def pixel(x, y):
    return [100 + 250 * x, 1000 - 250 * y]


def ladder(n=10, step=.35):
    """A climbable route, as detector polygons under FLAT."""
    return [square(*pixel(.35 if i % 2 == 0 else .75, .3 + step * i), 10) for i in range(n)]


class HoldFilterTests(unittest.TestCase):
    """What a calibration can say about a mask that is not a hold on this wall."""

    def place(self, calibration, polys, size):
        return measurement.holds_from_calibration(
            measurement.Grid(**calibration), polys, size)

    def test_places_a_hold_in_wall_metres(self):
        placed = self.place(FLAT, [square(*pixel(1, 1), 10)], FLAT_SIZE)
        self.assertEqual((placed.ids, placed.dropped), ([0], []))
        self.assertAlmostEqual(placed.holds[0].x, 1, places=3)
        self.assertAlmostEqual(placed.holds[0].y, 1, places=3)
        self.assertAlmostEqual(placed.holds[0].size, .08, places=3)

    def test_drops_masks_no_hold_could_be(self):
        polys = [square(*pixel(1, 1), 10), square(*pixel(1, 1), 200),
                 square(*pixel(1, 1), 1), np.array([[350., 750.], [360., 750.]])]
        placed = self.place(FLAT, polys, FLAT_SIZE)
        self.assertEqual(placed.ids, [0])
        reasons = {d['id']: d['reason'] for d in placed.dropped}
        self.assertIn('spans 1.60 m', reasons[1])  # a sign, not a hold
        self.assertIn('spans 0.01 m', reasons[2])  # mask noise
        self.assertIn('no area', reasons[3])

    def test_drops_what_the_vanishing_line_ruins(self):
        polys = [square(500, 400, 8), square(500, 900, 8), square(500, 990, 8)]
        placed = self.place(TILTED, polys, TILTED_SIZE)
        self.assertEqual(placed.ids, [0])
        reasons = {d['id']: d['reason'] for d in placed.dropped}
        self.assertIn('off the calibrated wall', reasons[1])
        self.assertIn('off the wall plane', reasons[2])

    def test_a_reference_scale_keeps_everything_it_can_place(self):
        reference = measurement.Reference(method='reference', points=[[0, 0], [250, 0]], metres=1)
        placed = measurement.holds_from_calibration(
            reference, [square(*pixel(1, 1), 10)], FLAT_SIZE)
        self.assertEqual(placed.ids, [0])


class BetaTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)
        server._cache['beta'] = {'size': FLAT_SIZE, 'polys': ladder()}
        self.body = {'id': 'beta', 'calibration': FLAT, 'route': list(range(10)), 'height_m': 1.68}

    def tearDown(self):
        server._cache.pop('beta', None)

    def post(self, **changes):
        return self.client.post('/api/beta', json=dict(self.body, **changes))

    def test_no_implicit_scale(self):
        # An assumed wall height used to reach the search here, so a beta could
        # be computed from a scale /api/measure would have refused.
        body = dict(self.body); body.pop('calibration')
        self.assertEqual(self.client.post('/api/beta', json=body).status_code, 422)
        self.assertEqual(self.post(wall_height_m=4.5).status_code, 422)
        self.assertEqual(self.post(calibration={}).status_code, 422)

    def test_climbs_a_projected_ladder(self):
        result = self.post().json()
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['off_wall'], [])
        self.assertIn('0.2 m', result['basis'])
        self.assertTrue(all(m['limb'] in ('LH', 'RH', 'LF', 'RF') for m in result['moves']))
        self.assertTrue(all(0 <= m['to'] < 10 for m in result['moves']))  # detector ids
        self.assertIn(9, (result['moves'][-1]['to'], *result['start'].values()))

    def test_a_route_hold_off_the_wall_is_explained(self):
        server._cache['beta']['polys'] = ladder() + [square(*pixel(1, 1), 200)]
        result = self.post(route=[0, 1, 2, 10])
        self.assertEqual(result.status_code, 200)
        self.assertFalse(result.json()['ok'])
        self.assertIn('[10]', result.json()['reason'])
        self.assertEqual([d['id'] for d in result.json()['off_wall']], [10])

    def test_implausible_calibration_is_refused_not_climbed(self):
        # Calling each 0.2 m gap 0.5 m stretches the ladder over 7.9 m of wall.
        result = self.post(calibration=dict(FLAT, spacing_m=.5)).json()
        self.assertFalse(result['ok'])
        self.assertIn('not a boulder', result['reason'])
        # Stretched far enough, the holds leave the calibrated block first.
        result = self.post(calibration=dict(FLAT, spacing_m=2)).json()
        self.assertFalse(result['ok'])
        self.assertIn('not on the calibrated wall plane', result['reason'])

    def test_validates_route_and_climber(self):
        self.assertEqual(self.post(route=[0, 1]).status_code, 422)
        self.assertEqual(self.post(route=[0, 1, 99]).status_code, 400)
        self.assertEqual(self.post(route=[0, 1, -1]).status_code, 422)
        self.assertEqual(self.post(height_m=0).status_code, 422)
        self.assertEqual(self.post(height_m=3).status_code, 422)
        self.assertEqual(self.post(id='missing').status_code, 404)
