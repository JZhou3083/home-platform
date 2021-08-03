# Copyright (c) 2018, Simon Brodeur
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

from panda3d.core import LVector3f, TransformState, ClockObject, LVecBase3f, BitMask32, LVector4f

from home_platform.utils import Viewer, vec3ToNumpyArray
from home_platform.suncg import SunCgSceneLoader, loadModel
from home_platform.rendering import RgbRenderer
from home_platform.physics import Panda3dBulletPhysics

#####
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "tests", "data", "suncg")
SUNCG_DATA_DIR=TEST_SUNCG_DATA_DIR
#####
#try:
#    SUNCG_DATA_DIR = os.environ["SUNCG_DATA_DIR"]
#except KeyError:
#    print("Please set the environment variable SUNCG_DATA_DIR")
#    sys.exit(1)

CDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


class Agent(object):

    def __init__(self, scene, agentId, agentRadius=0.25):
        self.scene = scene
        self.agentId = agentId

        agentsNp = self.scene.scene.find('**/agents')
        agentNp = agentsNp.attachNewNode(agentId)
        agentNp.setTag('agent-id', agentId)
        scene.agents.append(agentNp)

        # Define a simple sphere model
        modelId = 'sphere-0'
#        modelFilename = os.path.join(CDIR, 'sphere.egg')
        modelFilename = os.path.join(CDIR, '77.bam')

        agentNp.setTag('model-id', modelId)
        model = loadModel(modelFilename)
        model.setColor(LVector4f(np.random.uniform(), np.random.uniform(), np.random.uniform(), 1.0))
        model.setName('model-' + os.path.basename(modelFilename))
        model.setTransform(TransformState.makeScale(agentRadius))
        model.reparentTo(agentNp)
        model.hide(BitMask32.allOn())

        # Calculate the center of this object
        minBounds, maxBounds = model.getTightBounds()
        centerPos = minBounds + (maxBounds - minBounds) / 2.0

        # Add offset transform to make position relative to the center
        agentNp.setTransform(TransformState.makePos(centerPos))
        model.setTransform(model.getTransform().compose(TransformState.makePos(-centerPos)))

        self.agentNp = agentNp
        self.model = model
        self.agentRbNp = None

        self.rotationStepCounter = -1
        self.rotationsStepDuration = 40

    def _getAgentNode(self):
        if self.agentRbNp is None:
            agentRbNp = self.agentNp.find('**/+BulletRigidBodyNode')
            if agentRbNp is None:
                raise Exception(
                    'Unable to find the BulletRigidBodyNode instance related to the agent: the agent should be created before the physic engine!')
            self.agentRbNp = agentRbNp
        return self.agentRbNp

    def getPosition(self):
        agentRbNp = self._getAgentNode()
        return vec3ToNumpyArray(agentRbNp.getNetTransform().getPos())

    def getOrientation(self):
        agentRbNp = self._getAgentNode()
        return vec3ToNumpyArray(agentRbNp.getNetTransform().getHpr())

    def setPosition(self, position):
        agentRbNp = self._getAgentNode()
        agentRbNp.setPos(LVector3f(position[0], position[1], position[2]))

    def setOrientation(self, orientation):
        agentRbNp = self._getAgentNode()
        agentRbNp.setHpr(
            LVector3f(orientation[0], orientation[1], orientation[2]))

    def setLinearVelocity(self, linearVelocity):
        # Apply the local transform to the velocity
        agentRbNp = self._getAgentNode()
        rotMat = agentRbNp.node().getTransform().getMat().getUpper3()
        linearVelocity = rotMat.xformVec(LVecBase3f(
            linearVelocity[0], linearVelocity[1], linearVelocity[2]))
        linearVelocity.z = 0.0
        agentRbNp.node().setLinearVelocity(linearVelocity)
        agentRbNp.node().setActive(True, 1)

    def setAngularVelocity(self, angularVelocity):
        agentRbNp = self._getAgentNode()
        agentRbNp.node().setAngularVelocity(
            LVector3f(angularVelocity[0], angularVelocity[1], angularVelocity[2]))
        agentRbNp.node().setActive(True, 1)

    def step(self, observation):
        # TODO: do something useful with the observation
        x, y, z = observation['position']
        logger.info('Agent %s at position (x=%f, y=%f, z=%f)' % (self.agentId, x, y, z))

        # Constant speed forward (Y-axis)
        linearVelocity = LVector3f(0.0, 1.0, 0.0)
        self.setLinearVelocity(linearVelocity)

        # Randomly change angular velocity (rotation around Z-axis)
        if self.rotationStepCounter > self.rotationsStepDuration:
            # End of rotation
            self.rotationStepCounter = -1
            self.setAngularVelocity(np.zeros(3))
        elif self.rotationStepCounter >= 0:
            # During rotation
            self.rotationStepCounter += 1
        else:
            # No rotation, initiate at random
            if np.random.random() > 0.5:
                angularVelocity = np.zeros(3)
                angularVelocity[2] = np.random.uniform(low=-np.pi, high=np.pi)
                self.rotationStepCounter = 0
                self.setAngularVelocity(angularVelocity)


def main():

    # Create scene and remove any default agents
    scene = SunCgSceneLoader.loadHouseFromJson(houseId="0004d52d1aeeb8ae6de39d6bd993e992", datasetRoot=SUNCG_DATA_DIR)
    scene.scene.find('**/agents').node().removeAllChildren()
    scene.agents = []

    # Create multiple agents
    agents = []
    for i in range(3):
        agentRadius = 0.15
        agent = Agent(scene, 'agent-%d' % (i), agentRadius)
        agents.append(agent)

    # NOTE: specify to move the camera slightly outside the model (not to render the interior of the model)
    cameraTransform = TransformState.makePos(LVector3f(0.0, 1.1 * agentRadius, 0.0))

    # Initialize rendering and physics
    renderer = RgbRenderer(scene, size=(128, 128), fov=70.0, cameraTransform=cameraTransform)
    renderer.showRoomLayout(showCeilings=True, showWalls=True, showFloors=True)
    physics = Panda3dBulletPhysics(scene, SUNCG_DATA_DIR, objectMode='box',
                                   agentRadius=agentRadius, agentMode='sphere')

    # Configure the camera
    viewer = Viewer(scene, interactive=False, showPosition=False, cameraMask=renderer.cameraMask)
    transform = TransformState.makePosHpr(LVecBase3f(44.01, -43.95, 22.97),
                                          LVecBase3f(0.0, -81.04, 0.0))
    viewer.cam.setTransform(transform)

    # Initialize the agent
    agents[0].setPosition((45, -42.5, 1.6))
    agents[1].setPosition((42.5, -39, 1.6))
    agents[2].setPosition((42.5, -41.5, 1.6))

    # Initialize figure that will show the point-of-view of each agent
    plt.ion()
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ims = []
    for i in range(len(agents)):
        ax = fig.add_subplot(1, len(agents), i + 1)
        ax.set_title(agents[i].agentId)
        ax.axis('off')
        rgbImage = np.zeros(renderer.size + (3,), dtype=np.uint8)
        im = ax.imshow(rgbImage)
        ims.append(im)
    plt.tight_layout()
    plt.show()

    # Main loop
    clock = ClockObject.getGlobalClock()
    try:
        while True:

            # Update physics
            dt = clock.getDt()
            physics.step(dt)

            # Update viewer
            viewer.step()

            for i, agent in enumerate(agents):
                # Get the current RGB
                rgbImage = renderer.getRgbImage(
                    agent.agentId, channelOrder="RGB")

                # Get the current observation for the agent
                observation = {"position": agent.getPosition(),
                               "orientation": agent.getOrientation(),
                               "rgb-image": rgbImage}
                agent.step(observation)

                ims[i].set_data(rgbImage)

            fig.canvas.draw()
            plt.pause(0.0001)

    except KeyboardInterrupt:
        pass

    viewer.destroy()
    renderer.destroy()

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
