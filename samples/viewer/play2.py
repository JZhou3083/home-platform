# Copyright (c) 2017, IGLU consortium
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

from panda3d.core import LVector3f, TransformState, ClockObject, LVecBase3f, BitMask32, LVector4f,AudioSound

from home_platform.suncg import SunCgSceneLoader,loadModel
from home_platform.env import BasicEnvironment,myEnvironment
from home_platform.utils import Viewer,gamer
dataRoot = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "tests", "data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "tests", "data", "suncg")

logger = logging.getLogger(__name__)

# class Agent(object):
#
#     def __init__(self, scene, agentId, agentRadius=0.25):
#         self.scene = scene
#         self.agentId = agentId
#
#         agentsNp = self.scene.scene.find('**/agents')
#         agentNp = agentsNp.attachNewNode(agentId)
#         agentNp.setTag('agent-id', agentId)
#         scene.agents.append(agentNp)
#
#         # Define a model
#         modelId = 'sphere-0'
#         modelFilename = os.path.join(CDIR, 'sphere.egg')
#
#
#
#         agentNp.setTag('model-id', modelId)
#
#         model = Actor("models/eve",  # Load our animated charachter
#                          {'walk': "models/eve_walk"})
#
# #        model.setColor(LVector4f(np.random.uniform(), np.random.uniform(), np.random.uniform(), 1.0))
#         model.setColor(LVector4f(0.75,0.70,0.8, 1.0))
#         model.setName('model-' + os.path.basename(modelFilename))
#         model.setTransform(TransformState.makeScale(agentRadius))
#         model.reparentTo(agentNp)
#         model.hide(BitMask32.allOn())
#
#         eveNeck=model.controlJoint(None,'modelRoot','Neck')
#         eveNeck.setName('Neck')
#         eveNeck.hide(BitMask32.allOn())
#
#
#         # Calculate the center of this object
#         minBounds, maxBounds = model.getTightBounds()
#         centerPos = minBounds + (maxBounds - minBounds)
# #        centerPos[1]=centerPos[1]+1.0
#
#         # Add offset transform to make position relative to the center
#         agentNp.setTransform(TransformState.makePos(centerPos))
#         model.setTransform(model.getTransform().compose(TransformState.makePos(-centerPos)))
#
#
#
#         self.agentNp = agentNp
#         self.model = model
# 	self.eveNeck= eveNeck
#         self.agentRbNp = None
#
#
#         self.rotationStepCounter = -1
#         self.rotationsStepDuration = 40
#
#
#     def getName(self):
#     	return os.path.basename(os.path.join(CDIR, 'sphere.egg'))
#
#
#
#     def _getAgentNode(self):
#         if self.agentRbNp is None:
#             agentRbNp = self.agentNp.find('**/+BulletRigidBodyNode')
#             if agentRbNp is None:
#                 raise Exception(
#                     'Unable to find the BulletRigidBodyNode instance related to the agent: the agent should be created before the physic engine!')
#             self.agentRbNp = agentRbNp
#         return self.agentRbNp
#
#
#     def getPosition(self):
#         agentRbNp = self._getAgentNode()
#         return vec3ToNumpyArray(agentRbNp.getNetTransform().getPos())
#
#     def getOrientation(self):
#         agentRbNp = self._getAgentNode()
#         return vec3ToNumpyArray(agentRbNp.getNetTransform().getHpr())
#
#     def setPosition(self, position):
#         agentRbNp = self._getAgentNode()
#         agentRbNp.setPos(LVector3f(position[0], position[1], position[2]))
#         self.agentNp.setPos(LVector3f(position[0], position[1], position[2]))
#
#     def setOrientation(self, orientation):
#         agentRbNp = self._getAgentNode()
#         agentRbNp.setHpr(
#             LVector3f(orientation[0], orientation[1], orientation[2]))
#         self.agentNp.setHpr( LVector3f(orientation[0], orientation[1], orientation[2]))
#
#
#     def setLinearVelocity(self, linearVelocity):
#         # Apply the local transform to the velocity
#         agentRbNp = self._getAgentNode()
#         rotMat = agentRbNp.node().getTransform().getMat().getUpper3()
#         linearVelocity = rotMat.xformVec(LVecBase3f(
#             linearVelocity[0], linearVelocity[1], linearVelocity[2]))
#         linearVelocity.z = 0.0
#         agentRbNp.node().setLinearVelocity(linearVelocity)
#         agentRbNp.node().setActive(True, 1)
#
#     def setAngularVelocity(self, angularVelocity):
#         agentRbNp = self._getAgentNode()
#         agentRbNp.node().setAngularVelocity(
#             LVector3f(angularVelocity[0], angularVelocity[1], angularVelocity[2]))
# #        agentRbNp.node().setActive(True, 1)
#     def step(self, observation):
#         # TODO: do something useful with the observation
#         x, y, z = observation['position']
# #        logger.info('Agent %s at position (x=%f, y=%f, z=%f)' % (self.agentId, x, y, z))
#         logger.info('Agent %s at position (x=%f, y=%f, z=%f)' % (self.agentId, x, y, z))
#
#         # Constant speed forward (Y-axis)
#         linearVelocity = LVector3f(0.0, -0, 0.0)
#         # linearVelocity = LVector3f(1.0, 1.0, 0.0)
#         self.setLinearVelocity(linearVelocity)
#
#         # Randomly change angular velocity (rotation around Z-axis)
#         if self.rotationStepCounter > self.rotationsStepDuration:
#             # End of rotation
#             self.rotationStepCounter = -1
#             self.setAngularVelocity(np.zeros(3))
#         elif self.rotationStepCounter >= 0:
#             # During rotation
#             self.rotationStepCounter += 1
#         else:
#             # No rotation, initiate at random
#             if np.random.random() > 0.5:
#                 angularVelocity = np.zeros(3)
# #                angularVelocity[0] = np.random.uniform(low=-np.pi, high=np.pi) # the agent roate on x axis
# #                angularVelocity[1] = np.random.uniform(low=-np.pi, high=np.pi) # the agent roate on y axis
# #                angularVelocity[2] = np.random.uniform(low=-np.pi, high=np.pi) # the agent roate on z-axis
#
#                 self.rotationStepCounter = 0
#                 self.setAngularVelocity(angularVelocity)
class SoundSource(object):
    def __init__(self, scene, modelId, sourceSize=0.25):
        self.scene = scene
        self.modelId = modelId

        # Define a sound source
        sourceSize = 0.25
        modelId = 'source-0'
        modelFilename = os.path.join(dataRoot, 'models', 'sphere.egg')
        objectsNp = scene.scene.attachNewNode('objects')
        objectsNp.setTag('acoustics-mode', 'source')
        objectNp = objectsNp.attachNewNode('sound-' + modelId)
        model = loadModel(modelFilename)
        model.setName('model-' + modelId)
        model.setTransform(TransformState.makeScale(sourceSize))
        model.reparentTo(objectNp)
        # objectNp.setPos(LVecBase3f(39, -40.5, 1.5))

        objectNp.setPos(LVecBase3f(39, -40.5, 1.5))
        self.objectNp=objectNp
def main():
    
    env = myEnvironment(houseId="0004d52d1aeeb8ae6de39d6bd993e992", suncgDatasetRoot=TEST_SUNCG_DATA_DIR, realtime=True)
    print(env.scene.scene.ls())
    # env.setAgentOrientation((60.0, 0.0, 0.0))
    env.setAgentPosition((42, -39, 1.0))
    env.renderWorld.showRoomLayout(showCeilings=False, showWalls=True, showFloors=True)

    # Configure the camera
    cam_mode = 1 # 0 mode is god view, 1 is First person view
    viewer = gamer(env.scene, showPosition=True, cam_mode=cam_mode)
    if cam_mode:
        transform = TransformState.makePosHpr(LVecBase3f(0,-0.3, 0),
                                         LVecBase3f(180, 0, 0))
    else:
        transform = TransformState.makePosHpr(LVecBase3f(44.01, -43.95, 10),
                                          LVecBase3f(0.0, -81.04, 0.0))
    
    # # Find agent and reparent camera to it
    # agent = env.scene.scene.find('**/agents/agent*/+BulletRigidBodyNode')
    # viewer.camera.reparentTo(agent)
    viewer.cam.setTransform(transform)
    
    linearVelocity = np.zeros(3)
    angularVelocity = np.zeros(3)
    rotationStepCounter = -1 
    rotationsStepDuration = 40
    try:
        while True:
            
            # # Constant speed forward (Y-axis)
            # linearVelocity = LVector3f(0.0, 0.0, 0.0)
            # env.setAgentLinearVelocity(linearVelocity)
            #
            # # Randomly change angular velocity (rotation around Z-axis)
            # if rotationStepCounter > rotationsStepDuration:
            #     # End of rotation
            #     rotationStepCounter = -1
            #     angularVelocity = np.zeros(3)
            # elif rotationStepCounter >= 0:
            #     # During rotation
            #     rotationStepCounter += 1
            # else:
            #     # No rotation, initiate at random
            #     if np.random.random() > 0.5:
            #         angularVelocity = np.zeros(3)
            #         angularVelocity[2] = np.random.uniform(low=-np.pi, high=np.pi)
            #         rotationStepCounter = 0
            # env.setAgentAngularVelocity(angularVelocity)
            
            # Simulate
            env.step()
            print(env.scene.agents[0].getPos())

            viewer.step()
            
    except KeyboardInterrupt:
        pass

    viewer.destroy()
    env.destroy()


    return 0

if __name__ == "__main__":
    sys.exit(main())
