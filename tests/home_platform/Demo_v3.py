### changed the environment as CR2 room
import os
import csv
import time
import logging
import numpy as np
import unittest
import scipy
import matplotlib.pyplot as plt
from panda3d.core import TransformState, LVecBase3f, LMatrix4f, BitMask32, AudioSound,NodePath
from panda3d.core import LVector3f, TransformState, ClockObject, LVecBase3f, BitMask32, LVector4f,AudioSound
from direct.actor.Actor import Actor
from home_platform.realRoom import SunCgSceneLoader, loadModel
from home_platform.utils import Viewer_jz, vec3ToNumpyArray
from pyroomacoustics.soundsource import SoundSource
from scipy.io import wavfile
from home_platform.acoustics_pra import EvertAcoustics_jz, CipicHRTF,\
    MaterialAbsorptionTable, AirAttenuationTable, AudioPlayer,\
    interauralPolarToVerticalPolarCoordinates,\
    verticalPolarToInterauralPolarCoordinates,\
    verticalPolarToCipicCoordinates
from home_platform.rendering_CR2 import RgbRenderer
from home_platform.physics import Panda3dBulletPhysics_jz
TEST_DATA_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "data")
modelpath= os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "data", "models")
TEST_REAL_DATA_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "data", "RealRoom")

### locations

LsPos1 = LVecBase3f(0.931,-2.547,1.23)
LsPos2 = LVecBase3f(0.119,2.88,1.23)

mp1 = LVecBase3f(-0.993,1.426,1.23)
mp2 = LVecBase3f(0.439,-0.147,1.23)
mp3 = LVecBase3f(1.361,-0.603,1.23)

mp4 = LVecBase3f(-1.11,-0.256,1.23)
mp5 = LVecBase3f(-0.998,-1.409,1.230)


class TestEvertAcoustics(unittest.TestCase):
    def testRenderHouseWithAcousticsPath(self):

        scene = SunCgSceneLoader.loadHouseFromJson("CR2", TEST_REAL_DATA_DIR)
        # print(scene.scene.ls())
        agentId = 5 # 1-5


        sourceId = 1 #1-2


        agenID = [mp1,mp2,mp3,mp4,mp5]
        srcID = [LsPos1,LsPos2]
        agentNp = scene.agents[0]
        model = Actor(os.path.join(TEST_DATA_DIR,"models",'miro-e.bam'))  # Load our animated charachter)
        model.setColor(LVector4f(np.random.uniform(), np.random.uniform(), np.random.uniform(), 1.0))
        # model.setColor(LVector4f(0.3,0.3,0.6, 1.0))
        model.setTransform(TransformState.makeScale(1))
        # eveNeck=model.controlJoint(None,'modelRoot','Neck')
        model.reparentTo(agentNp)
        agentNp.setPos(agenID[agentId-1])
        agentNp.setHpr(-90, 0, 0)






        # Define a sound source
        srcPosition=srcID[sourceId-1]
        srcOrientation = LVecBase3f(0, 0, 0)
        sourceSize = 0.15
        modelId = 'source-0'
        modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'sphere.egg')
        objectsNp = scene.scene.attachNewNode('objects')
        objectsNp.setTag('acoustics-mode', 'source')
        objectNp = objectsNp.attachNewNode('object-' + modelId)
        model = loadModel(modelFilename)
        model.setName('model-' + modelId)
        model.setTransform(TransformState.makeScale(sourceSize))
        model.reparentTo(objectNp)

        objectNp.setPos(srcPosition)
        objectNp.setHpr(srcOrientation)
        samplingRate = 44100.0





        # hrtf = CipicHRTF(os.path.join(TEST_DATA_DIR, 'hrtf',
        #
        #                               microtransform = [TransformState.makePos(LVecBase3f(0.15,0.03, 0.6)),TransformState.makePos(LVecBase3f(-0.15,0.03, 0.6))] ## two microphones on the ears'cipic_hrir.mat'), samplingRate) ## unused
        #microtransform = [TransformState.makePos(LVecBase3f(-0.08,0.06, 0.2)),TransformState.makePos(LVecBase3f(-0.08,-0.06, 0.2))] ## third microphone on the tail ,TransformState.makePos(LVecBase3f(0,-0.15, 0.6))
        #microtransform = [TransformState.makePos(LVecBase3f(0.15,0.03, 0.6)),TransformState.makePos(LVecBase3f(-0.15,0.03, 0.6))] ## two microphones on the ears
        microtransform = [TransformState.makePos(LVecBase3f(0, 0, 0))]
        orderNb = 1
        SpeakerId = sourceId
        MicroId = agentId
        acoustics = EvertAcoustics_jz(
            scene, None, samplingRate, maximumOrder=2, microphoneTransform=microtransform, ray_tracing=True,
        maxBufferLength=30)

        acoustics.showRoomLayout(showCeilings=False)

        # Attach sound to object
        filename = os.path.join(TEST_DATA_DIR, 'audio', 'music.wav')
        fs,audio = wavfile.read(filename)


        sound = SoundSource(srcPosition,signal = audio)
        acoustics.attachSoundToObject(sound, objectNp)
        sound.setLoop(True)
        sound.setLoopCount(3)
        sound.play()
        acoustics.step(0.1)

        #### This is for debugging and IR plotting
        # acoustics.world.image_source_model()
        # acoustics.world.ray_tracing()
        # acoustics.world.simulate()
        # imp = acoustics.world.rir[0][0]

        # imp= acoustics.calculateImpulseResponse("source-0","agent-0")

        # plot of two microphones impulses
        # fig = plt.figure()
        # data = imp
        # t = np.linspace(0,len(data)/samplingRate,len(data))
        # plt.title("unsampled impulse response of Beam Tracing")
        # plt.xlabel("TIme(s)")
        # plt.plot(t,data)
        # acoustics.world.plot_rir()
        # plt.show()
        # fileNa= "IR_LS" + str(SpeakerId) + "_MP"+str(MicroId)
        # plt.xlabel('Time(s)')
        # # plt.ylabel('s')
        # plt.show(block=False)
        # plt.title("Impulse response,LS1:(0.931,-2.547,1.23),Mic5:(-0.998,-1.409,1.230)")
        # time.sleep(10.0)
        # plt.savefig(fileNa, bbox_inches='tight')
        # plt.close(fig)
        # wavfile.write(str(fileNa)+".wav",samplingRate,data)


        ## initial estimation of the Reverberation Time
        # rt60 = acoustics.world.measure_rt60(plot=True)
        # plt.show()
        # print(acoustics.world.rir[0])
        from pyroomacoustics.experimental import measure_rt60
        # # acoustics.world.plot_rir(FD=True) ## enery curve ploting
        # print(rt60)



        physics = Panda3dBulletPhysics_jz(scene, TEST_REAL_DATA_DIR, objectMode='box',agentRadius=0.15, agentMode='sphere')

        renderer = RgbRenderer(scene, size=(128, 128), fov=70.0, cameraTransform=None)
        renderer.showRoomLayout(showCeilings=False, showWalls=True, showFloors=True)
        viewer = Viewer_jz(scene,nbMicrophones=2, interactive=True,showPosition=False)
        # Configure the camera
        # NOTE: in Panda3D, the X axis points to the right, the Y axis is
        # forward, and Z is up
        center = agentNp.getNetTransform().getPos()
        transform = TransformState.makePosHpr(LVecBase3f(center.x, center.y-7, 12.5),
                                          LVecBase3f(0.0, -50, 0.0))
        viewer.cam.setTransform(transform)


        # impulse response
        imp = []
        jing = []
        jing.append(0)
        Store_flag = True
        time_cur = 0

        k = 0
        # main loop
        clock = ClockObject.getGlobalClock()

        try:
            while True:
                # update physics
                dt = clock.getDt()
                print(dt)
                physics.step(dt)
                acoustics.step(dt)

                # Update viewer
                viewer.step()

        #
        except KeyboardInterrupt:

            pass

        viewer.destroy()
        physics.destroy()
        viewer.graphicsEngine.removeAllWindows()

        # Calculate and show impulse responses
        renderer.destroy()
        acoustics.destroy()


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
