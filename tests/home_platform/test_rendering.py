# Copyright (c) 2017, IGLU consortium
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
import time
import logging
import numpy as np
import unittest
import matplotlib.pyplot as plt

from panda3d.core import LMatrix4f, TransformState, LVecBase3, BitMask32, LVector3f

from home_platform.rendering import Panda3dRenderer, Panda3dSemanticsRenderer, InstancesRenderer, RgbRenderer, DepthRenderer
from home_platform.suncg import SunCgSceneLoader, loadModel, SunCgModelLights
from home_platform.core import Scene
from home_platform.utils import Viewer
from home_platform.constants import MODEL_CATEGORY_COLOR_MAPPING

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "suncg")

try:
    SUNCG_DATA_DIR = os.environ["SUNCG_DATA_DIR"]
except KeyError:
    raise Exception("Please set the environment variable SUNCG_DATA_DIR")

class TestPanda3dRenderer(unittest.TestCase):
    def testObjectWithViewer(self):

        scene = Scene()

        modelId = '83'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        model = loadModel(modelFilename)
        model.setName('model-' + str(modelId))
        model.show(BitMask32.allOn())

        objectsNp = scene.scene.attachNewNode('objects')
        objNp = objectsNp.attachNewNode('object-' + str(modelId))
        model.reparentTo(objNp)

        # Calculate the center of this object
        minBounds, maxBounds = model.getTightBounds()
        centerPos = minBounds + (maxBounds - minBounds) / 2.0

        # Add offset transform to make position relative to the center
        model.setTransform(TransformState.makePos(-centerPos))

        renderer = None
        viewer = None

        try:
            renderer = Panda3dRenderer(scene, shadowing=False)

            viewer = Viewer(scene, interactive=False)
            viewer.disableMouse()

            viewer.cam.setTransform(TransformState.makePos(LVecBase3(5.0, 0.0, 0.0)))
            viewer.cam.lookAt(model)

            for _ in range(20):
                viewer.step()
            time.sleep(1.0)

        finally:
            if renderer is not None:
                renderer.destroy()
            if viewer is not None:
                viewer.destroy()
                viewer.graphicsEngine.removeAllWindows()

    def testStep(self):

        scene = SunCgSceneLoader.loadHouseFromJson("0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)

        modelLightsInfo = SunCgModelLights(os.path.join(TEST_SUNCG_DATA_DIR, 'metadata', 'suncgModelLights.json'))
        renderer = Panda3dRenderer(scene, shadowing=True, mode='offscreen', modelLightsInfo=modelLightsInfo)
        renderer.showRoomLayout(showCeilings=False)

        mat = np.array([0.999992, 0.00394238, 0, 0,
                        -0.00295702, 0.750104, -0.661314, 0,
                        -0.00260737, 0.661308, 0.75011, 0,
                        43.621, -55.7499, 12.9722, 1])
        scene.agents[0].setMat(LMatrix4f(*mat.ravel()))

        renderer.step(dt=0.1)
        image = renderer.getRgbImages()['agent-0']
        depth = renderer.getDepthImages(mode='distance')['agent-0']
        self.assertTrue(np.min(depth) >= renderer.zNear)
        self.assertTrue(np.max(depth) <= renderer.zFar)

        fig = plt.figure(figsize=(16, 8))
        plt.axis("off")
        ax = plt.subplot(121)
        ax.imshow(image)
        ax = plt.subplot(122)
        ax.imshow(depth / np.max(depth), cmap='binary')
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)

        renderer.destroy()


class TestPanda3dSemanticsRenderer(unittest.TestCase):
    def testStep(self):

        scene = SunCgSceneLoader.loadHouseFromJson("0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)

        renderer = Panda3dSemanticsRenderer(scene, TEST_SUNCG_DATA_DIR, mode='offscreen')
        renderer.showRoomLayout(showCeilings=False)

        mat = np.array([0.999992, 0.00394238, 0, 0,
                        -0.00295702, 0.750104, -0.661314, 0,
                        -0.00260737, 0.661308, 0.75011, 0,
                        43.621, -55.7499, 12.9722, 1])
        scene.agents[0].setMat(LMatrix4f(*mat.ravel()))

        renderer.step(dt=0.1)
        image = renderer.getRgbaImages()['agent-0']

        # Validate that all rendered colors maps to original values, up to some tolerance
        eps = 1e-2
        colors = np.stack(MODEL_CATEGORY_COLOR_MAPPING.values())
        for color in image.reshape((-1, image.shape[-1])):
            alpha = color[-1]
            if alpha == 255:
                self.assertTrue(np.min(np.sum(np.abs(colors - color[:3]), axis=1)) < eps)

        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ax = plt.subplot(111)
        ax.imshow(image)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)

        renderer.destroy()

#### Commented due to acbsence of house files in "SUNCG_DATA_DIR"
#class TestInstancesRenderer(unittest.TestCase):

#    def testGetVisibleObjectIds(self):

#        houseId = "0004d52d1aeeb8ae6de39d6bd993e992"
#        scene = SunCgSceneLoader.loadHouseFromJson(houseId, SUNCG_DATA_DIR)
#        agent = scene.agents[0]

#        # Configure the agent
#        transform = TransformState.makePosHpr(pos=LVector3f(38.42, -39.10, 1.70),
#                                              hpr=LVector3f(-77.88, -13.93, 0.0))
#        agent.setTransform(transform)

#        renderer = InstancesRenderer(scene, size=(512, 512), fov=75.0)

#        agentId = agent.getTag('agent-id')
#        image = renderer.getInstancesImage(agentId)

#        fig = plt.figure(figsize=(8, 8))
#        plt.ion()
#        plt.show()
#        plt.axis("off")
#        plt.imshow(image)

#        plt.draw()
#        plt.pause(1.0)
#        plt.close(fig)

#        visibleObjectIds = renderer.getVisibleObjectIds(agentId)
#        self.assertTrue(len(visibleObjectIds) == 28)
####

class TestRgbRenderer(unittest.TestCase):

    def testGetRgbImage(self):

        houseId = "0004d52d1aeeb8ae6de39d6bd993e992"
        scene = SunCgSceneLoader.loadHouseFromJson(houseId, SUNCG_DATA_DIR)
        agent = scene.agents[0]

        # Configure the agent
        transform = TransformState.makePosHpr(pos=LVector3f(38.42, -39.10, 1.70),
                                              hpr=LVector3f(-77.88, -13.93, 0.0))
        agent.setTransform(transform)

        renderer = RgbRenderer(scene, size=(512, 512), fov=75.0)

        agentId = agent.getTag('agent-id')
        image = renderer.getRgbImage(agentId)

        fig = plt.figure(figsize=(8, 8))
        plt.ion()
        plt.show()
        plt.axis("off")
        plt.imshow(image)

        plt.draw()
        plt.pause(1.0)
        plt.close(fig)


class TestDepthRenderer(unittest.TestCase):

    def testInit(self):

        houseId = "0004d52d1aeeb8ae6de39d6bd993e992"
        scene = SunCgSceneLoader.loadHouseFromJson(houseId, SUNCG_DATA_DIR)
        agent = scene.agents[0]

        # Configure the agent
        transform = TransformState.makePosHpr(pos=LVector3f(38.42, -39.10, 1.70),
                                              hpr=LVector3f(-77.88, -13.93, 0.0))
        agent.setTransform(transform)

        renderer = DepthRenderer(scene, size=(512, 512), fov=75.0)

        agentId = agent.getTag('agent-id')
        image = renderer.getDepthImage(agentId)

        fig = plt.figure(figsize=(8, 8))
        plt.ion()
        plt.show()
        plt.axis("off")
        plt.imshow(image)

        plt.draw()
        plt.pause(1.0)
        plt.close(fig)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
