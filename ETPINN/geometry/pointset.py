import numpy as np

from .geometry import Geometry
from ..data import BatchSampler


class PointSet(Geometry):
    """A geometry defined by a set of points.

    Args:
        points: A NumPy array of shape (`N`, `dx`). A list of `dx`-dim points.
    """

    def __init__(self, points):
        self.points = np.asarray(points)
        self.num_points = len(points)
        super().__init__(
            len(points[0]),
            (np.amin(self.points, axis=0), np.amax(self.points, axis=0)),
            np.inf,
        )

        self.sampler = BatchSampler(self.num_points, shuffle=True)

    def inside(self, x):
        raise NotImplementedError("dde.geometry.PointSet doesn't support inside.")

    def on_boundary(self, x):
        raise NotImplementedError("dde.geometry.PointSet doesn't support on_boundary.")

    def random_points(self, n, random="pseudo"):
        if n <= self.num_points:
            indices = self.sampler.get_next(n)
            return self.points[indices]

        x = np.tile(self.points, (n // self.num_points, 1))
        indices = self.sampler.get_next(n % self.num_points)
        return np.vstack((x, self.points[indices]))

    def random_boundary_points(self, n, random="pseudo"):
        raise NotImplementedError(
            "dde.geometry.PointSet doesn't support random_boundary_points."
        )
