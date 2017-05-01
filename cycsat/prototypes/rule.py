"""
rule.py

A library of default rule definitions. Rules need a "run" function that returns the
a mask of location it can be placed.
"""
from cycsat.archetypes import Rule
from shapely.ops import cascaded_union
from shapely.geometry import Point, LineString
from shapely.affinity import translate
from shapely.affinity import rotate, scale

import random


class OUTSIDE(Rule):
    """Allows a observable to appear outside the site bounds."""
    __mapper_args__ = {'polymorphic_identity': 'OUTSIDE'}

    def __init__(self, value=10000):
        """Returns a Feature by "placing it."""
        self.kind = 'restrict'
        self.pattern = None
        self.value = value

    def run(self, Simulator, **params):
        return self.observable.site.bounds().buffer(int(self.value))


class NEAR(Rule):
    __mapper_args__ = {'polymorphic_identity': 'NEAR'}

    def __init__(self, pattern, value=100):
        """Returns a Feature by "placing it."""
        self.kind = 'restrict'
        self.pattern = pattern
        self.value = value

    def run(self, Simulator, **params):
        # get targets
        targets = self.depends_on(Simulator)['obj'].tolist()
        if not targets:
            return self.observable.site.bounds()
        mask = cascaded_union(
            [target.footprint(timestep=params['timestep']) for target in targets])

        inner_buffer = mask.buffer(int(self.value))
        mask = self.observable.footprint().bounds

        diagaonal_dist = Point(mask[0:2]).distance(Point(mask[2:]))

        buffer_value = diagaonal_dist * 2
        second_buffer = inner_buffer.buffer(buffer_value)
        mask = second_buffer.difference(inner_buffer)

        return mask


class WITHIN(Rule):
    __mapper_args__ = {'polymorphic_identity': 'WITHIN'}

    def __init__(self, pattern, value=0, kind='restrict'):
        """Returns a Feature by "placing it."""
        self.kind = kind
        self.pattern = pattern
        self.value = value

    def run(self, Simulator, **params):
        # get targets
        targets = self.depends_on(Simulator)['obj'].tolist()
        if not targets:
            return self.observable.site.bounds()
        mask = cascaded_union(
            [target.footprint(timestep=params['timestep']) for target in targets])

        return mask.buffer(int(self.value))


class XALIGN(Rule):
    __mapper_args__ = {'polymorphic_identity': 'XALIGN'}

    def __init__(self, pattern=None, value=None):
        """Returns a Feature by "placing it."""
        self.kind = 'place'
        self.pattern = pattern
        self.value = value

    def run(self, Simulator, **params):
        # get targets
        maxy = self.observable.site.maxy
        if self.value:
            line = LineString([[int(self.value), 0], [int(self.value), maxy]])

        else:
            targets = self.depends_on(Simulator)['obj'].tolist()
            if not targets:
                return self.observable.site.bounds()

            mask = cascaded_union(
                [target.footprint(timestep=params['timestep']) for target in targets])

            value = mask.centroid.x
            line = LineString([[value, 0], [value, maxy]])

        return line.buffer(10)


class YALIGN(Rule):
    __mapper_args__ = {'polymorphic_identity': 'YALIGN'}

    def __init__(self, pattern=None, value=None):
        """Returns a Feature by "placing it."""
        self.kind = 'place'
        self.pattern = pattern
        self.value = value

    def run(self, Simulator, **params):
        # get targets
        maxx = self.observable.site.maxx
        if self.value:
            line = LineString([[0, int(self.value)], [maxx, int(self.value)]])

        else:
            targets = self.depends_on(Simulator)['obj'].tolist()
            if not targets:
                return self.observable.site.bounds()

            mask = cascaded_union(
                [target.footprint(timestep=params['timestep']) for target in targets])

            value = mask.centroid.y
            line = LineString([[0, value], [maxx, value]])

        return line.buffer(10)


class ROTATE(Rule):
    __mapper_args__ = {'polymorphic_identity': 'ROTATE'}

    def __init__(self, pattern=None, value='random'):
        """Returns a Feature by "placing it."""
        self.kind = 'transform'
        self.pattern = pattern
        self.value = value

    def run(self, Simulator, **params):
        # if there are any targets get the first angle
        angle = self.rotation = random.randint(-180, 180)
        try:
            angle = int(self.value)
        except:
            if self.pattern:
                targets = self.depends_on(Simulator)['obj'].tolist()
                if targets:
                    angle = targets[0].rotation

        self.observable.rotate(angle, params['timestep'])


class DISPURSE_PLUME(Rule):
    __mapper_args__ = {'polymorphic_identity': 'DISPURSE_PLUME'}

    def __init__(self, pattern=None):
        """Returns a Feature by "placing it."""
        self.kind = 'transform'
        self.pattern = pattern
        self.value = None

    def run(self, Simulator, **params):

        wind_dir = random.randint(30, 80)

        for shape in self.observable.shapes:

            geometry = shape.geometry(params['timestep'])
            oval = scale(geometry, 1, 1.5)
            oval = translate(oval, 0, -600)
            roval = rotate(oval, wind_dir, origin=geometry.centroid)

            shape.add_loc(params['timestep'],
                          wkt=roval.wkt)
