import os
import time
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

from panda3d.core import LVector3f, TransformState, ClockObject, LVecBase3f, BitMask32, LVector4f,AudioSound


from home_platform.utils import gamer, vec3ToNumpyArray
from home_platform.suncg import SunCgSceneLoader, loadModel
from home_platform.rendering import RgbRenderer
from home_platform.physics import Panda3dBulletPhysics
from home_platform.core import Scene


from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.core import TextNode, NodePath, LightAttrib,ModelNode
from panda3d.core import LVector3
from direct.actor.Actor import Actor
from direct.task.Task import Task
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.DirectObject import DirectObject
import sys


### These are for the audio
from home_platform.acoustics import EvertAcoustics, CipicHRTF, FilterBank, \
    MaterialAbsorptionTable, AirAttenuationTable, EvertAudioSound, AudioPlayer,\
    interauralPolarToVerticalPolarCoordinates,\
    verticalPolarToInterauralPolarCoordinates,\
    verticalPolarToCipicCoordinates


#####
CDIR = os.path.dirname(os.path.realpath(__file__)) # loacte the current file path
TEST_SUNCG_DATA_DIR = os.path.join(CDIR, "..", "..", "tests", "data", "suncg")
SUNCG_DATA_DIR=TEST_SUNCG_DATA_DIR
TEST_DATA_DIR = os.path.join(CDIR, "..", "..", "tests", "data")
#####
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
        modelFilename = os.path.join(CDIR, 'sphere.egg')



        agentNp.setTag('model-id', modelId)

        model = Actor("models/eve",  # Load our animated charachter
                         {'walk': "models/eve_walk"})

#        model.setColor(LVector4f(np.random.uniform(), np.random.uniform(), np.random.uniform(), 1.0))
        model.setColor(LVector4f(0.75,0.70,0.8, 1.0))
        model.setName('model-' + os.path.basename(modelFilename))
        model.setTransform(TransformState.makeScale(agentRadius))
        model.reparentTo(agentNp)
        model.hide(BitMask32.allOn())

        eveNeck=model.controlJoint(None,'modelRoot','Neck')
        eveNeck.setName('Neck')
        eveNeck.hide(BitMask32.allOn())


        # Calculate the center of this object
        minBounds, maxBounds = model.getTightBounds()
        centerPos = minBounds + (maxBounds - minBounds)
#        centerPos[1]=centerPos[1]+1.0

        # Add offset transform to make position relative to the center
        agentNp.setTransform(TransformState.makePos(centerPos))
        model.setTransform(model.getTransform().compose(TransformState.makePos(-centerPos)))



        self.agentNp = agentNp
        self.model = model
	self.eveNeck= eveNeck
        self.agentRbNp = None


        self.rotationStepCounter = -1
        self.rotationsStepDuration = 40


    def getName(self):
    	return os.path.basename(os.path.join(CDIR, 'sphere.egg'))



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
#        agentRbNp.node().setActive(True, 1)
    def step(self, observation):
        # TODO: do something useful with the observation
        x, y, z = observation['position']
#        logger.info('Agent %s at position (x=%f, y=%f, z=%f)' % (self.agentId, x, y, z))
        logger.info('Agent %s at position (x=%f, y=%f, z=%f)' % (self.agentId, x, y, z))

        # Constant speed forward (Y-axis)
        linearVelocity = LVector3f(0.0, -0, 0.0)
        # linearVelocity = LVector3f(1.0, 1.0, 0.0)
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
#                angularVelocity[0] = np.random.uniform(low=-np.pi, high=np.pi) # the agent roate on x axis
#                angularVelocity[1] = np.random.uniform(low=-np.pi, high=np.pi) # the agent roate on y axis
#                angularVelocity[2] = np.random.uniform(low=-np.pi, high=np.pi) # the agent roate on z-axis

                self.rotationStepCounter = 0
                self.setAngularVelocity(angularVelocity)
def main():
    scene = SunCgSceneLoader.loadHouseFromJson(
            "0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)
    ### THis part specify how many agents you need
    scene.agents = []
    # Create multiple agents
    agents = []
    for i in range(1):
        agentRadius = 0.15
        agent = Agent(scene, 'agent-%d' % (i), agentRadius)
        agents.append(agent)
    agentNp = scene.agents[0]
    agentNp.setPos(LVecBase3f(45, -42.5, 1.6))
    agentNp.setHpr(45, 0, 0)
    # # Define a sound source
    # sourceSize = 0.25
    # modelId = 'source-0'
    # modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'sphere.egg')
    # objectsNp = scene.scene.attachNewNode('objects')
    # objectsNp.setTag('acoustics-mode', 'source')
    # objectNp = objectsNp.attachNewNode('sound-' + modelId)
    # model = loadModel(modelFilename)
    # model.setName('model-' + modelId)
    # model.setTransform(TransformState.makeScale(sourceSize))
    # model.reparentTo(objectNp)
    # objectNp.setPos(LVecBase3f(39, -40.5, 1.5))
    # agent.agentNp.setPos(LVecBase3f(40, -40.5, 1.5))

    # Define multiple sound sources
    sources = []
    for i, pos in enumerate([(39, -40.5, 1.5), (45.5, -42.5, 0.5)]):
        sourceSize = 0.25
        modelId = 'source-%d' % (i)
        modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'sphere.egg')
        objectsNp = scene.scene.attachNewNode('objects')
        objectsNp.setTag('acoustics-mode', 'source')
        objectNp = objectsNp.attachNewNode('object-' + modelId)
        model = loadModel(modelFilename)
        model.setName('model-' + modelId)
        model.setTransform(TransformState.makeScale(sourceSize))
        model.reparentTo(objectNp)
        objectNp.setPos(LVecBase3f(*pos))
        sources.append(objectNp)

    samplingRate = 16000.0
    hrtf = CipicHRTF(os.path.join(TEST_DATA_DIR, 'hrtf',
                                      'cipic_hrir.mat'), samplingRate)
    acoustics = EvertAcoustics(
            scene, hrtf, samplingRate, maximumOrder=2, debug=True, maxBufferLength=30.0)
    # physics = Panda3dBulletPhysics(scene, SUNCG_DATA_DIR, objectMode='box',
    #                               agentRadius=0.3, agentMode='sphere')

    # # Attach sound to object
    # filename = os.path.join(TEST_DATA_DIR, 'audio', 'toilet.ogg')
    # sound = EvertAudioSound(filename)
    #
    # acoustics.attachSoundToObject(sound, objectNp)

    audioFilenames = ['toilet.ogg', 'radio.ogg']
    sounds = []
    for audioFilename, source in zip(audioFilenames, sources):
        # Attach sound to object
        filename = os.path.join(TEST_DATA_DIR, 'audio', audioFilename)
        sound = EvertAudioSound(filename)
        acoustics.attachSoundToObject(sound, source)
        sound.setLoop(True)
        sound.setLoopCount(1)

        sounds.append(sound)
    sounds[0].play()
    acoustics.step(dt=5.0)
    sounds[0].stop()
    sounds[1].play()
    acoustics.step(dt=5.0)
    sounds[0].play()
    sounds[1].play()
    acoustics.step(dt=5.0)
    # sound.play()

    # acoustics.step(0.0)
    # physics.step(0.0)
    # Calculate and show impulse responses
    impulse = acoustics.calculateImpulseResponse(
            objectNp.getName(), str(scene.agents[0].getName()))
    obs = acoustics.getObservationsForAgent(scene.agents[0].getName())
    data = np.array([obs['audio-buffer-left'],
                         obs['audio-buffer-right']], dtype=np.float32).T
    print(data)
    fig = plt.figure()
    plt.plot(data)
    plt.show(block=False)
    time.sleep(2.0)
    plt.close(fig)



    fig = plt.figure()
    plt.plot(impulse.impulse[0], color='b', label='Left channel')
    plt.plot(impulse.impulse[1], color='g', label='Right channel')
    plt.legend()
    plt.show(block=False)
    time.sleep(2.0)
    plt.close(fig)






    # acoustics.destroy()
    # physics.destroy()



    return 0


if __name__ == "__main__":
#    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
