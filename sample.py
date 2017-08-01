from cycsat.agent import Agent
from cycsat.rules import NEAR, ALIGN
from cycsat.geometry import grid
from cycsat.laboratory import Material

from shapely.geometry import Polygon, box, Point
import random
import matplotlib.pyplot as plt


class CoolingTowerBlock(Agent):

    def __init__(self, **variables):
        Agent.__init__(self, **variables)


class CoolingTower(Agent):

    def __init__(self, **variables):
        Agent.__init__(self, **variables)

    def __run__(self):
        if random.choice([True, False]):
            self.on = 1
            self.value += 1
            print('on')
        else:
            print('off')
            self.on = 0
        return True


class Plume(Agent):

    def __init__(self, **variables):
        Agent.__init__(self, name='plume', **variables)

    def __run__(self):

        if self.parent.on == 1:
            self.place_in(self.parent.relative_geo.buffer(100))
        else:
            self.geometry = None

site = Agent(geometry=box(0, 0, 1000, 1000), name='site', value=100)

cblock = CoolingTowerBlock(geometry=box(0, 0, 500, 500), value=10)
cblock.add_rules(NEAR('CoolingTower1', 'CoolingTower', value=50))
cblock.add_rules(ALIGN('CoolingTower1', 'CoolingTower', axis='x'))
cblock.add_rules(ALIGN('turbine', 'CoolingTower1', axis='y'))

turbine = Agent(name='turbine', geometry=box(0, 0, 50, 100))
ctower1 = CoolingTower(on=0, geometry=Point(0, 0).buffer(75), value=20)
ctower2 = CoolingTower(on=0, geometry=Point(0, 0).buffer(75), value=20)
plume = Plume(geometry=Point(0, 0).buffer(50), value=100)

cblock.add_agents([ctower1, ctower2, turbine])
ctower1.add_agent(plume)
site.add_agent(cblock)

# 25 build tests!
fig, axes = plt.subplots(5, 5)
axes = axes.flatten()

for ax in axes:
    ax.set_aspect('equal')
    site.place()
    site.agenttree.plot(ax=ax)
