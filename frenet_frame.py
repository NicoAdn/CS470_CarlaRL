import numpy as np
import sys

import sys
import glob
import os

try:
    sys.path.append(glob.glob('./carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
 


class FrenetFrame:
    """save waypoint in form of frenet frame for retrieving frenet frame coordinate.\n
    coordinate : (s,t), we
    s : longitudunal distance along path
    t : latitudinal distance from path"""
    def __init__(self):
        """waypoints : list of carla.Location"""

        # list of np.array([x, y]). (x, y) is location of waypt
        self.waypoints = None

        # longitudinal distance of each waypoint from 1st waypoint
        self.long_dist = None

    def set_waypoints(self, waypoints):
        """
        waypoints : list of carla.Waypoint or carla.Location
        """
        if isinstance(waypoints[0], carla.Waypoint):
            self.waypoints = [np.array([waypt.transform.location.x, waypt.transform.location.y]) 
                              for waypt in waypoints]

        elif isinstance(waypoints[0], carla.Location):
            self.waypoints = [np.array([waypt.x, waypt.y]) 
                              for waypt in waypoints]
        else:
            raise ValueError("only list of carla.Waypoint or carla.Location is allowed")

        self.long_dist = []
        self.calculate_longitudinal()

    def calculate_longitudinal(self):
        """calculate longitudinal distance from 1st waypoint"""

        for i, waypt in enumerate(self.waypoints):
            if i == 0:
                self.long_dist.append(0)
            else:
                pre_waypt = self.waypoints[i-1]
                cur_waypt = self.waypoints[i]

                long_dist = np.linalg.norm(cur_waypt - pre_waypt) + self.long_dist[i-1]

                self.long_dist.append(long_dist)


    def get_distance(self, location):
        """calculate frenet frame coordinate (s,t) of a location
        location : Carla.Location
        return : (s, t)
        s : longitudinal coordinate
        t : latitudinal coordinate"""

        location = np.array([location.x, location.y])
        start_idx, end_idx = self.get_segment(location)
        start = self.waypoints[start_idx]
        end = self.waypoints[end_idx]

        s2l = location - start
        s2e = end - start

        # s : longitudinal
        # projection of a vector start waypt to location on a vector start to end waypt
        proj = np.dot(s2l, s2e) / np.dot(s2e, s2e)* s2e

        


        # the sign of longitidunal difference from start
        sign = np.sign(np.dot(s2l, s2e))
        s = sign * np.linalg.norm(proj) + self.long_dist[start_idx]

        # t: latitudinal
        normal = s2l - proj

        # the sign of t
        sign = -np.sign(np.cross(np.append(s2l, 0), np.append(s2e, 0))[-1])
        t = sign * np.linalg.norm(normal)

        return s, t

    def get_closest(self, location):
        """get index of closet waypoint from a location
        location : np.array([x, y])"""

        squared_distances = np.sum((np.array(self.waypoints) - location) ** 2, axis = 1)

        return np.argmin(squared_distances)

    def get_segment(self, location):
        """get segment where the location belongs to. 
        location : np.array([x, y])
        return segment (idx1, idx2) : idx of start and end location. 
        """
        
        closest_idx = self.get_closest(location)

        if closest_idx == 0:
            return (closest_idx, closest_idx + 1)
        elif closest_idx == len(self.waypoints) - 1:
            return (closest_idx - 1, closest_idx)

        closest = self.waypoints[closest_idx]
        next = self.waypoints[closest_idx + 1]

        c2l = location - closest
        c2n = next - closest

        if np.dot(c2l, c2n) >= 0:
            return (closest_idx, closest_idx + 1)
        else:
            return (closest_idx - 1, closest_idx)

if __name__ == "__main__":
    waypts = [carla.Location(0, 0, 0), carla.Location(2, 2, 0), carla.Location(4, 2, 0), carla.Location(7, 2, 0), carla.Location(9, 0, 0)]
    location = carla.Location(-2, 0, 0)

    FF = FrenetFrame()
    FF.set_waypoints(waypts)
    print("Result: ", FF.get_distance(location))

