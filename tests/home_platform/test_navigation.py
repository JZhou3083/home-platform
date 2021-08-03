# Copyright (c) 2018, Simon Brodeur
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os
import logging
import unittest
import numpy as np
import matplotlib.pyplot as plt

from home_platform.navigation import NavigationHelper, getRegionLabeledOccupacyMap, NavigationGraph
from home_platform.suncg import SunCgSceneLoader

try:
    SUNCG_DATA_DIR = os.environ["SUNCG_DATA_DIR"]
except KeyError:
    raise Exception("Please set the environment variable SUNCG_DATA_DIR")

TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "suncg")

#### added this to make it running with test data
SUNCG_DATA_DIR=TEST_SUNCG_DATA_DIR
####

logger = logging.getLogger(__name__)


class TestFunctions(unittest.TestCase):

    def testGetRegionLabeledOccupacyMap(self):

        houseId = "0004d52d1aeeb8ae6de39d6bd993e992"
        scene = SunCgSceneLoader.loadHouseFromJson(houseId, SUNCG_DATA_DIR)

        helper = NavigationHelper(scene)
        occupancyMap, _, _ = helper.calculateOccupancyMap(resolution=0.1)

        labeledOccupancyMap = getRegionLabeledOccupacyMap(
            occupancyMap)
        nbRegions = int(np.max(labeledOccupancyMap))
        self.assertTrue(np.array_equal(
            labeledOccupancyMap.shape, occupancyMap.shape))
        self.assertTrue(nbRegions == 4)

        # Colorize the map randomly
        image = np.zeros((labeledOccupancyMap.shape[0],
                          labeledOccupancyMap.shape[1],
                          3))
        for r in range(1, nbRegions + 1):
            randomColor = np.random.uniform(size=(3,))
            image[labeledOccupancyMap == r] = randomColor

        fig = plt.figure(figsize=(8, 8))
        plt.ion()
        plt.show()
        plt.axis("off")

        plt.imshow(image)

        plt.draw()
        plt.pause(5.0)
        plt.close(fig)


class TestNavigationGraph(unittest.TestCase):

    def testInit(self):
        nodes = [np.array([0.0, 0.0, 0.0]),
                 np.array([0.0, 1.0, 0.0]),
                 np.array([0.0, 1.0, 1.0])]
        connectivity = [[1, 2],
                        [0],
                        [1]]
        NavigationGraph(nodes, connectivity)

    def testToNx(self):

        nodes = [np.array([0.0, 0.0, 0.0]),
                 np.array([0.0, 1.0, 0.0]),
                 np.array([0.0, 1.0, 1.0])]
        connectivity = [[1, 2],
                        [1],
                        [2]]
        graph = NavigationGraph(nodes, connectivity)
        graph = graph.toNx()
        self.assertTrue(graph.number_of_nodes() == 3)
        self.assertTrue(graph.number_of_edges() == 4)


class TestNavigationHelper(unittest.TestCase):

    def testCalculateWallMap(self):

        houseId = "0004d52d1aeeb8ae6de39d6bd993e992"
        scene = SunCgSceneLoader.loadHouseFromJson(houseId, SUNCG_DATA_DIR)

        helper = NavigationHelper(scene)
        wallMap, xlim, ylim = helper.calculateWallMap(resolution=0.1)
        self.assertTrue(wallMap.shape[0] == wallMap.shape[1])
        self.assertTrue(wallMap.ndim == 2)

        factorX = wallMap.shape[0] / \
            (xlim[1] - xlim[0])  # pixel per meter
        factorY = wallMap.shape[1] / \
            (ylim[1] - ylim[0])  # pixel per meter
        self.assertTrue(np.allclose(factorX, factorY, atol=1e-6))

        fig = plt.figure(figsize=(8, 8))
        plt.ion()
        plt.show()
        plt.axis("off")

        plt.imshow(wallMap, cmap='gray')

        plt.draw()
        plt.pause(5.0)
        plt.close(fig)

    def testCalculateFloorMap(self):

        houseId = "0004d52d1aeeb8ae6de39d6bd993e992"
        scene = SunCgSceneLoader.loadHouseFromJson(houseId, SUNCG_DATA_DIR)

        helper = NavigationHelper(scene)
        floorMap, xlim, ylim = helper.calculateFloorMap(resolution=0.1)
        self.assertTrue(floorMap.shape[0] == floorMap.shape[1])
        self.assertTrue(floorMap.ndim == 2)

        factorX = floorMap.shape[0] / \
            (xlim[1] - xlim[0])  # pixel per meter
        factorY = floorMap.shape[1] / \
            (ylim[1] - ylim[0])  # pixel per meter
        self.assertTrue(np.allclose(factorX, factorY, atol=1e-6))

        fig = plt.figure(figsize=(8, 8))
        plt.ion()
        plt.show()
        plt.axis("off")

        plt.imshow(floorMap, cmap='gray')

        plt.draw()
        plt.pause(5.0)
        plt.close(fig)

    def testCalculateObstacleMap(self):

        houseId = "0004d52d1aeeb8ae6de39d6bd993e992"
        scene = SunCgSceneLoader.loadHouseFromJson(houseId, SUNCG_DATA_DIR)

        helper = NavigationHelper(scene)
        obstacleMap, xlim, ylim = helper.calculateObstacleMap(resolution=0.1)
        self.assertTrue(obstacleMap.shape[0] == obstacleMap.shape[1])
        self.assertTrue(obstacleMap.ndim == 2)

        factorX = obstacleMap.shape[0] / \
            (xlim[1] - xlim[0])  # pixel per meter
        factorY = obstacleMap.shape[1] / \
            (ylim[1] - ylim[0])  # pixel per meter
        self.assertTrue(np.allclose(factorX, factorY, atol=1e-6))

        fig = plt.figure(figsize=(8, 8))
        plt.ion()
        plt.show()
        plt.axis("off")

        plt.imshow(obstacleMap, cmap='gray')

        plt.draw()
        plt.pause(5.0)
        plt.close(fig)

    def testCalculateOccupancyMap(self):

        houseId = "0004d52d1aeeb8ae6de39d6bd993e992"
        scene = SunCgSceneLoader.loadHouseFromJson(houseId, SUNCG_DATA_DIR)

        helper = NavigationHelper(scene)
        occupancyMap, xlim, ylim = helper.calculateOccupancyMap(resolution=0.1)
        self.assertTrue(occupancyMap.shape[0] == occupancyMap.shape[1])
        self.assertTrue(occupancyMap.ndim == 2)

        factorX = occupancyMap.shape[0] / \
            (xlim[1] - xlim[0])  # pixel per meter
        factorY = occupancyMap.shape[1] / \
            (ylim[1] - ylim[0])  # pixel per meter
        self.assertTrue(np.allclose(factorX, factorY, atol=1e-6))

        fig = plt.figure(figsize=(8, 8))
        plt.ion()
        plt.show()
        plt.axis("off")

        plt.imshow(occupancyMap, cmap='gray')

        plt.draw()
        plt.pause(5.0)
        plt.close(fig)

    def testCalculateNavigationGraph(self):

        houseId = "0004d52d1aeeb8ae6de39d6bd993e992"
        scene = SunCgSceneLoader.loadHouseFromJson(houseId, SUNCG_DATA_DIR)

        helper = NavigationHelper(scene)
        graph, occupancyMap, xlim, ylim = helper.calculateNavigationGraph(resolution=0.1,
                                                                          level=0, safetyMarginEdges=0.10)

        fig = plt.figure(figsize=(10, 10))
        plt.ion()
        plt.show()

        plt.imshow(occupancyMap, cmap='gray', origin='upper',
                   extent=[xlim[0], xlim[1], ylim[0], ylim[1]])

        for i, ps in enumerate(graph.nodes):
            for k in graph.connectivity[i]:
                psk = graph.nodes[k]
                plt.plot([ps[0], psk[0]], [ps[1], psk[1]], 'green', zorder=1)

            plt.plot(ps[0], ps[1], 'r.', zorder=2)

        plt.draw()
        plt.pause(5.0)
        plt.close(fig)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    unittest.main()
