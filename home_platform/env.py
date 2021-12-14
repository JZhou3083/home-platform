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
#  - Neither the name of the NECOTIS research group nor the names of its contributors
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

import heapq
import logging
import os
import numpy as np

from home_platform.suncg import SunCgSceneLoader,loadModel
from home_platform.rendering import Panda3dRenderer, Panda3dSemanticsRenderer
from home_platform.physics import Panda3dBulletPhysics
from home_platform.acoustics import EvertAcoustics

from direct.actor.Actor import Actor
from panda3d.core import ClockObject, LVector3f, TransformState, LVecBase3f,BitMask32,LVector4f
from home_platform.utils import vec3ToNumpyArray

from home_platform.acoustics import EvertAcoustics, CipicHRTF, FilterBank, \
    MaterialAbsorptionTable, AirAttenuationTable, EvertAudioSound, AudioPlayer,\
    interauralPolarToVerticalPolarCoordinates,\
    verticalPolarToInterauralPolarCoordinates,\
    verticalPolarToCipicCoordinates
dataRoot = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "tests", "data")
logger = logging.getLogger(__name__)


def extractAllRegions(occupacyGrid):
    occupacyGrid = np.atleast_2d(occupacyGrid).astype(np.int)
    occupacyGrid[occupacyGrid != 0] = -1

    curRegionId = 1
    while np.count_nonzero(occupacyGrid == 0) > 0:

        # Apply Grassfire algorithm (also known as Wavefront or Brushfire
        # algorithm)
        heap = []
        heapq.heapify(heap)
        visited = set()

        # Select a start point that is not yet labelled, add the heap queue
        start = tuple(np.argwhere(occupacyGrid == 0)[0])
        heapq.heappush(heap, start)
        visited.add(start)

        while len(heap) > 0:
            # Get cell from heap queue, assign to current region and add to
            # visited set
            i, j = heapq.heappop(heap)
            occupacyGrid[i, j] = curRegionId

            # Add all 4 neighbors to heap queue, if not occupied
            for ai, aj in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:

                # Check bounds
                if (ai >= 0 and ai < occupacyGrid.shape[0] and aj >= 0 and aj < occupacyGrid.shape[1]):
                    # Check occupancy and redundancy
                    if occupacyGrid[ai, aj] == 0 and (ai, aj) not in visited:
                        heapq.heappush(heap, (ai, aj))
                        visited.add((ai, aj))

        curRegionId += 1

    assert np.all(occupacyGrid != 0)
    logger.debug('Number of regions found in occupancy map: %d' %
                 (curRegionId - 1))

    return occupacyGrid


class Observation(object):
    def __init__(self, position, orientation, image, collision):
        self.position = position
        self.orientation = orientation
        self.image = image
        self.collision = collision

    def as_dict(self):
        return self.__dict__
class Agent(object):

    def __init__(self, scene, agentId, agentRadius=0.25):
        self.scene = scene
        self.agentId = agentId

        agentsNp = self.scene.scene.find('**/agents')
        agentNp = agentsNp.attachNewNode(agentId)
        agentNp.setTag('agent-id', agentId)
        scene.agents.append(agentNp)

        # Define a model
        modelId = 'Ralph'
        modelFilename = os.path.join(dataRoot, 'models','eve')
        agentNp.setTag('model-id', modelId)
        model = Actor(modelFilename,  # Load our animated charachter
                         {'walk': "models/eve_walk"})

#        model.setColor(LVector4f(np.random.uniform(), np.random.uniform(), np.random.uniform(), 1.0))
        model.setColor(LVector4f(0.75,0.70,0.8, 1.0))
        model.setName('model-' + os.path.basename(modelFilename))
        model.setTransform(TransformState.makeScale(agentRadius))
        model.reparentTo(agentNp)
        # model.hide(BitMask32.allOn())
        eveNeck=model.controlJoint(None,'modelRoot','Neck')
        eveNeck.setName('Neck')
        eveNeck.hide(BitMask32.allOn())
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

class myEnvironment(object):
    def __init__(self, houseId, suncgDatasetRoot=None, size=(256, 256), debug=False, depth=False, semantics=False, realtime=False, dt=0.1, cameraTransform=None):

        self.__dict__.update(houseId=houseId, suncgDatasetRoot=suncgDatasetRoot, size=size,
                             debug=debug, depth=depth, semantics=semantics, realtime=realtime, dt=dt, cameraTransform=cameraTransform)

        self.scene = SunCgSceneLoader.loadHouseFromJson(
            houseId, suncgDatasetRoot)
        self.scene.scene.find('**/agents').node().removeAllChildren()
         ### THis part specify how many agents you need
        self.scene.agents = []
        # Create multiple agents
        agents = []
        for i in range(1):
            agentRadius = 0.15
            agent = Agent(self.scene, 'agent-%d' % (i), agentRadius)
            agents.append(agent)
        agentHeight=1.6
        agentRadius=0.25
        sources=[]
        for j in range(1):
   	        source=SoundSource(self.scene,'-%d'%(j))
   	        sources.append(source)


        if self.cameraTransform is None:
            self.cameraTransform = TransformState.makePos(
                LVector3f(0.0, 0.0, agentHeight / 2.0 - agentRadius))
        self.renderWorld = Panda3dRenderer(
            self.scene, size, shadowing=False, depth=depth, cameraTransform=self.cameraTransform)

        self.physicWorld = Panda3dBulletPhysics(self.scene, suncgDatasetRoot, debug=debug, objectMode='box',
                                                agentRadius=agentRadius, agentHeight=agentHeight, agentMass=60.0, agentMode='capsule')
        samplingRate = 16000
        hrtf = CipicHRTF(os.path.join(suncgDatasetRoot,'..','hrtf',
                                      'cipic_hrir.mat'), samplingRate)
        self.acousticWorld = EvertAcoustics(self.scene, hrtf, samplingRate, maximumOrder=2, debug=True)
        self.clock = ClockObject.getGlobalClock()
        filename = os.path.join(dataRoot, 'audio', 'toilet.ogg')
        sound = EvertAudioSound(filename)
        self.acousticWorld.attachSoundToObject(sound, sources[0].objectNp)
        self.worlds = {
            "physics": self.physicWorld,
            "render": self.renderWorld,
            "acoustics": self.acousticWorld,
        }

        if semantics:
            self.renderSemanticWorld = Panda3dSemanticsRenderer(
                self.scene, suncgDatasetRoot, size, cameraTransform=self.cameraTransform)
            self.worlds["render-semantics"] = self.renderSemanticWorld

        self.agent = self.scene.agents[0]
        self.agentRbNp = self.agent.find('**/+BulletRigidBodyNode')
        self.Neck = self.agent.find(
            '**/+BulletRigidBodyNode/model*/Neck')

        self.labeledNavMap = None
        self.occupancyMapCoord = None

    def setAgentPosition(self, position):
        self.agentRbNp.setPos(LVector3f(position[0], position[1], position[2]))

    def setAgentOrientation(self, orientation):
        self.agentRbNp.setHpr(
            LVector3f(orientation[0], orientation[1], orientation[2]))
    def setHeadOrientation(self,orientation):
        self.Neck.setHpr(LVector3f(orientation[0], orientation[1], orientation[2]))
    def setAgentLinearVelocity(self, linearVelocity):
        # Apply the local transform to the velocity
        # XXX: use BulletCharacterControllerNode class, which already handles
        # local transform?
        rotMat = self.agentRbNp.node().getTransform().getMat().getUpper3()
        linearVelocity = rotMat.xformVec(LVecBase3f(
            linearVelocity[0], linearVelocity[1], linearVelocity[2]))
        linearVelocity.z = 0.0
        self.agentRbNp.node().setLinearVelocity(linearVelocity)
        self.agentRbNp.node().setActive(True, 1)

    def setAgentAngularVelocity(self, angularVelocity):
        self.agentRbNp.node().setAngularVelocity(
            LVector3f(angularVelocity[0], angularVelocity[1], angularVelocity[2]))
        self.agentRbNp.node().setActive(True, 1)

    def setHeadAngularVelocity(self, angularVelocity):
        self.Neck.node().setAngularVelocity(
            LVector3f(angularVelocity[0], angularVelocity[1], angularVelocity[2]))
        self.Neck.node().setActive(True, 1)

    def applyImpulseToAgent(self, impulse):
        self.agentRbNp.node().applyCentralImpulse(
            LVector3f(impulse[0], impulse[1], impulse[2]))

    def destroy(self):
        if self.renderWorld is not None:
            self.renderWorld.destroy()

    def generateOccupancyMap(self, minRegionArea=10.0, z=1.0, precision=0.1):

        cellArea = precision ** 2
        occupancyMap, self.occupancyMapCoord = self.physicWorld.calculate2dNavigationMap(self.agent, z=z,
                                                                                         precision=precision)
        self.labeledNavMap = extractAllRegions(occupancyMap)

        # Post-processing by removing regions smaller than the threshold
        curValidRegionId = 1
        nbRegions = int(np.max(self.labeledNavMap))
        regionAreas = []
        for r in range(1, nbRegions + 1):
            nbCells = np.count_nonzero(self.labeledNavMap == r)
            regionArea = nbCells * cellArea
            if regionArea < minRegionArea:
                self.labeledNavMap[self.labeledNavMap == r] = -1
            else:
                self.labeledNavMap[self.labeledNavMap == r] = curValidRegionId
                regionAreas.append(regionArea)
                curValidRegionId += 1

        assert np.all(self.labeledNavMap != 0)
        logger.debug('Number of valid regions found in occupancy map: %d' % (
            curValidRegionId - 1))

        return self.labeledNavMap, self.occupancyMapCoord

    def generateSpawnPositions(self, n, minRegionArea=10.0, z=1.0, precision=0.1):

        self.generateOccupancyMap(minRegionArea, z, precision)

        # Calculate region areas
        cellArea = precision ** 2
        nbRegions = int(np.max(self.labeledNavMap))
        regionAreas = []
        for r in range(1, nbRegions + 1):
            nbCells = np.count_nonzero(self.labeledNavMap == r)
            regionArea = nbCells * cellArea
            regionAreas.append(regionArea)
        regionAreas = np.array(regionAreas)

        # Calculate relative ratios between regions, and create a cummulative
        # array
        regionAreas /= np.sum(regionAreas)
        s = np.cumsum(regionAreas)

        positions = []
        for _ in range(n):

            if len(s) > 1:
                # Select a random region
                r = np.random.random()
                idx = np.argwhere(s < r)[-1] + 1
                selRegionId = idx + 1
            else:
                selRegionId = 1

            # Select a random cell of the region
            coord = np.argwhere(self.labeledNavMap == selRegionId)
            r = np.random.randint(low=0, high=len(coord))

            i, j = coord[r]
            x, y = self.occupancyMapCoord[i, j]
            positions.append([x, y])

        occupancyMap = np.zeros(self.labeledNavMap.shape, dtype=np.float)
        occupancyMap[self.labeledNavMap == -1] = 1.0

        return occupancyMap, self.occupancyMapCoord, np.array(positions)

    def getObservation(self):

        position = vec3ToNumpyArray(self.agentRbNp.getNetTransform().getPos())
        orientation = vec3ToNumpyArray(
            self.agentRbNp.getNetTransform().getHpr())
        image = self.renderWorld.getRgbImages()['agent-0']
        collision = self.physicWorld.isCollision(self.agentRbNp)

        return Observation(position, orientation, image, collision)
    def generateInpulse(self,source,agent):
        impulse = self.acousticWorld.calculateImpulseResponse(
            source.objectNp.getName(), agent.getName())
    def step(self):

        if self.realtime:
            dt = self.clock.getDt()
        else:
            dt = self.dt

        # NOTE: we should always update the physics first
        self.worlds["physics"].step(dt)
        self.worlds["render"].step(dt)

        if self.semantics:
            self.worlds["render-semantics"].step(dt)

class BasicEnvironment(object):
    def __init__(self, houseId, suncgDatasetRoot=None, size=(256, 256), debug=False, depth=False, semantics=False, realtime=False, dt=0.1, cameraTransform=None):

        self.__dict__.update(houseId=houseId, suncgDatasetRoot=suncgDatasetRoot, size=size,
                             debug=debug, depth=depth, semantics=semantics, realtime=realtime, dt=dt, cameraTransform=cameraTransform)

        self.scene = SunCgSceneLoader.loadHouseFromJson(
            houseId, suncgDatasetRoot)

        agentRadius = 0.1
        agentHeight = 1.6
        if self.cameraTransform is None:
            self.cameraTransform = TransformState.makePos(
                LVector3f(0.0, 0.0, agentHeight / 2.0 - agentRadius))
        self.renderWorld = Panda3dRenderer(
            self.scene, size, shadowing=False, depth=depth, cameraTransform=self.cameraTransform)

        self.physicWorld = Panda3dBulletPhysics(self.scene, suncgDatasetRoot, debug=debug, objectMode='box',
                                                agentRadius=agentRadius, agentHeight=agentHeight, agentMass=60.0, agentMode='capsule')

        self.clock = ClockObject.getGlobalClock()

        self.worlds = {
            "physics": self.physicWorld,
            "render": self.renderWorld,
        }

        if semantics:
            self.renderSemanticWorld = Panda3dSemanticsRenderer(
                self.scene, suncgDatasetRoot, size, cameraTransform=self.cameraTransform)
            self.worlds["render-semantics"] = self.renderSemanticWorld

        self.agent = self.scene.agents[0]
        self.agentRbNp = self.agent.find('**/+BulletRigidBodyNode')

        self.labeledNavMap = None
        self.occupancyMapCoord = None

    def setAgentPosition(self, position):
        self.agentRbNp.setPos(LVector3f(position[0], position[1], position[2]))

    def setAgentOrientation(self, orientation):
        self.agentRbNp.setHpr(
            LVector3f(orientation[0], orientation[1], orientation[2]))

    def setAgentLinearVelocity(self, linearVelocity):
        # Apply the local transform to the velocity
        # XXX: use BulletCharacterControllerNode class, which already handles
        # local transform?
        rotMat = self.agentRbNp.node().getTransform().getMat().getUpper3()
        linearVelocity = rotMat.xformVec(LVecBase3f(
            linearVelocity[0], linearVelocity[1], linearVelocity[2]))
        linearVelocity.z = 0.0
        self.agentRbNp.node().setLinearVelocity(linearVelocity)
        self.agentRbNp.node().setActive(True, 1)

    def setAgentAngularVelocity(self, angularVelocity):
        self.agentRbNp.node().setAngularVelocity(
            LVector3f(angularVelocity[0], angularVelocity[1], angularVelocity[2]))
        self.agentRbNp.node().setActive(True, 1)

    def applyImpulseToAgent(self, impulse):
        self.agentRbNp.node().applyCentralImpulse(
            LVector3f(impulse[0], impulse[1], impulse[2]))

    def destroy(self):
        if self.renderWorld is not None:
            self.renderWorld.destroy()

    def generateOccupancyMap(self, minRegionArea=10.0, z=1.0, precision=0.1):

        cellArea = precision ** 2
        occupancyMap, self.occupancyMapCoord = self.physicWorld.calculate2dNavigationMap(self.agent, z=z,
                                                                                         precision=precision)
        self.labeledNavMap = extractAllRegions(occupancyMap)

        # Post-processing by removing regions smaller than the threshold
        curValidRegionId = 1
        nbRegions = int(np.max(self.labeledNavMap))
        regionAreas = []
        for r in range(1, nbRegions + 1):
            nbCells = np.count_nonzero(self.labeledNavMap == r)
            regionArea = nbCells * cellArea
            if regionArea < minRegionArea:
                self.labeledNavMap[self.labeledNavMap == r] = -1
            else:
                self.labeledNavMap[self.labeledNavMap == r] = curValidRegionId
                regionAreas.append(regionArea)
                curValidRegionId += 1

        assert np.all(self.labeledNavMap != 0)
        logger.debug('Number of valid regions found in occupancy map: %d' % (
            curValidRegionId - 1))

        return self.labeledNavMap, self.occupancyMapCoord

    def generateSpawnPositions(self, n, minRegionArea=10.0, z=1.0, precision=0.1):

        self.generateOccupancyMap(minRegionArea, z, precision)

        # Calculate region areas
        cellArea = precision ** 2
        nbRegions = int(np.max(self.labeledNavMap))
        regionAreas = []
        for r in range(1, nbRegions + 1):
            nbCells = np.count_nonzero(self.labeledNavMap == r)
            regionArea = nbCells * cellArea
            regionAreas.append(regionArea)
        regionAreas = np.array(regionAreas)

        # Calculate relative ratios between regions, and create a cummulative
        # array
        regionAreas /= np.sum(regionAreas)
        s = np.cumsum(regionAreas)

        positions = []
        for _ in range(n):

            if len(s) > 1:
                # Select a random region
                r = np.random.random()
                idx = np.argwhere(s < r)[-1] + 1
                selRegionId = idx + 1
            else:
                selRegionId = 1

            # Select a random cell of the region
            coord = np.argwhere(self.labeledNavMap == selRegionId)
            r = np.random.randint(low=0, high=len(coord))

            i, j = coord[r]
            x, y = self.occupancyMapCoord[i, j]
            positions.append([x, y])

        occupancyMap = np.zeros(self.labeledNavMap.shape, dtype=np.float)
        occupancyMap[self.labeledNavMap == -1] = 1.0

        return occupancyMap, self.occupancyMapCoord, np.array(positions)

    def getObservation(self):

        position = vec3ToNumpyArray(self.agentRbNp.getNetTransform().getPos())
        orientation = vec3ToNumpyArray(
            self.agentRbNp.getNetTransform().getHpr())
        image = self.renderWorld.getRgbImages()['agent-0']
        collision = self.physicWorld.isCollision(self.agentRbNp)

        return Observation(position, orientation, image, collision)

    def step(self):

        if self.realtime:
            dt = self.clock.getDt()
        else:
            dt = self.dt

        # NOTE: we should always update the physics first
        self.worlds["physics"].step(dt)
        self.worlds["render"].step(dt)

        if self.semantics:
            self.worlds["render-semantics"].step(dt)
