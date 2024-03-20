# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import random
from itertools import combinations
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import Flow, Mission, Route, Scenario, Traffic, TrafficActor
from smarts.sstudio.types import Distribution, LaneChangingModel, JunctionModel

# normal = TrafficActor(
#     name="car",
# )

normal = TrafficActor(
    name="car",
    max_speed = 13.89,
    vehicle_type = 'passenger',
    speed = Distribution(mean=random.uniform(0.5, 1.0), sigma=0.0),
)

aggressive = TrafficActor(
    name="car",
    max_speed = 13.89,
    depart_speed = 'max',
    vehicle_type = 'passenger',
    lane_changing_model = LaneChangingModel(impatience=0.05, cooperative=0.05,
                                            pushy=0.05),
    speed = Distribution(mean=random.uniform(0.5, 1.0), sigma=0.0),
)
'''
route1 = ["edge-east-EW",0, "edge-south-ES",0, "edge-south-NS",0]
route2 = ["edge-east-EW",0, "edge-south-ES",0, "edge-west-SW",0, "edge-west-EW", 0]
'''

route1 = [("edge-east-EW",0, "edge-south-NS",0),
          ("edge-east-EW",0, "edge-west-EW", 0),
          ("edge-east-EW",0, "edge-north-SN",0),
          ("edge-east-EW",0, "edge-east-WE",0)]

route2 = [("edge-north-NS",0, "edge-south-NS",0),
          ("edge-north-NS",0, "edge-west-EW", 0),
          ("edge-north-NS",0, "edge-north-SN",0),
          ("edge-north-NS",0, "edge-east-WE",0)]

route3 = [("edge-west-WE",0, "edge-south-NS",0),
          ("edge-west-WE",0, "edge-west-EW", 0),
          ("edge-west-WE",0, "edge-north-SN",0),
          ("edge-west-WE",0, "edge-east-WE",0)]

route4 = [("edge-south-SN",0, "edge-south-NS",0),
          ("edge-south-SN",0, "edge-west-EW", 0),
          ("edge-south-SN",0, "edge-north-SN",0),
          ("edge-south-SN",0, "edge-east-WE",0)]

all_routes = route1 + route2 + route3 + route4
route_comb = [com for elems in range(1, 5) for com in combinations(all_routes, elems)]
traffic = {}
for name, routes in enumerate(route_comb):
    traffic[str(name)] = Traffic(
        engine="SUMO",
        flows=[
            Flow(
                route=Route(
                    begin=(start_edge, start_lane, 0),
                    end=(end_edge, end_lane, "max"),
                ),
                # Random flow rate, between x and y vehicles per minute.
                #rate=60 * random.uniform(10, 12),
                rate=100,
                # Random flow start time, between x and y seconds.
                #begin=random.uniform(0, 3),
                begin=0,
                # For an episode with maximum_episode_steps=3000 and step
                # time=0.1s, the maximum episode time=300s. Hence, traffic is
                # set to end at 900s, which is greater than maximum episode
                # time of 300s.
                end=60 * 15,
                # actors={aggressive: 1.0}
                actors={normal: 0.5, aggressive: 1.0},
            )
            for start_edge, start_lane, end_edge, end_lane in routes
        ]
    )

route = Route(begin=("edge-east-EW", 0, "max"), end=("edge-west-EW", 0, "max"))
ego_missions = [
    Mission(
        route=route,
        start_time=4  # Delayed start, to ensure road has prior traffic.
    )
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
    ),
    output_dir=Path(__file__).parent,
)
'''
    begin: Tuple[str, int, Any]
    """The (road, lane_index, offset) details of the start location for the route.

    road:
        The starting road by name.
    lane_index:
        The lane index from the rightmost lane.
    offset:
        The offset in metres into the lane. Also acceptable\\: "max", "random"
    """
    '''
