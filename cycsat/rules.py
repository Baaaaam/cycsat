from shapely.ops import nearest_points
from shapely.geometry import LineString, Point, box
from .geometry import calulate_shift


class Rule:

    def __init__(self, target, dep, **args):
        self.__target__ = target
        self.__dep__ = dep
        self.agent = False
        self.args = args

        for arg in args:
            setattr(self, arg, args[arg])

    def __repr__(self):
        return '<"{}" {} "{}" {}>'.format(self.__target__,
                                          self.__class__.__name__,
                                          self.__dep__,
                                          self.args)

    @property
    def target(self):
        try:
            return [i for i in self.agent.agents if i.name == self.__target__][0]
        except:
            return False

    @property
    def depend(self):
        try:
            if self.__dep__ == '$parent$':
                if self.target.parent:
                    return self.target.parent
        except:
            pass
        try:
            return [i for i in self.agent.agents if i.name == self.__dep__][0]
        except:
            return False

    def evaluate(self):
        try:
            return self.__evaluate__()
        except BaseException as e:
            print(e)
            return False

    def sharpen(self):
        try:
            return self.__sharpen__()
        except BaseException as e:
            print(e)
            return False


class SET(Rule):

    def __evaluate__(self):
        parent = self.depend.relative_geo
        minx, miny, maxx, maxy = parent.bounds
        parent_bounds = box(minx, miny, maxx, maxy)

        shift_x = (maxx - minx) * self.x
        shift_y = (maxy - miny) * self.y
        print(shift_x, shift_y)

        center_x, center_y = parent_bounds.centroid.x, parent_bounds.centroid.y
        center_x += shift_x
        center_y += shift_y

        return Point(center_x, center_y).buffer(self.padding)


class NEAR(Rule):

    def __evaluate__(self):
        inner_buffer = self.depend.geometry.buffer(self.value)
        outer_buffer = inner_buffer.buffer(100)
        return outer_buffer.difference(inner_buffer)

    # def __sharpen__(self):
    #     inner_buffer = self.depend.geometry.buffer(self.args['value'])
    #     x, y = nearest_points(self.target.geometry, inner_buffer)
    #     xoff, yoff = calulate_shift(x, y)
    #     self.target.move(xoff, yoff)


class ALIGN(Rule):

    def __evaluate__(self):

        value = getattr(self.depend.geometry.centroid, self.axis)

        if self.axis == 'x':
            maxy = self.target.parent.geometry.bounds[-1]
            line = LineString([[value, 0], [value, maxy]])
        else:
            maxx = self.target.parent.geometry.bounds[-1]
            line = LineString([[0, value], [maxx, value]])

        return line.buffer(10)
