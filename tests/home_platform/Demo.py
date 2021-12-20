import os
import csv
import time
import logging
import numpy as np
import unittest
import matplotlib.pyplot as plt
from panda3d.core import TransformState, LVecBase3f, LMatrix4f, BitMask32, AudioSound
from panda3d.core import LVector3f, TransformState, ClockObject, LVecBase3f, BitMask32, LVector4f,AudioSound
from direct.actor.Actor import Actor
from home_platform.suncg import SunCgSceneLoader, loadModel
from home_platform.core import Scene
from home_platform.utils import Viewer_jz, vec3ToNumpyArray
from home_platform.acoustics import EvertAcoustics, EvertAcoustics_jz,CipicHRTF, FilterBank, \
    MaterialAbsorptionTable, AirAttenuationTable, EvertAudioSound, AudioPlayer,\
    interauralPolarToVerticalPolarCoordinates,\
    verticalPolarToInterauralPolarCoordinates,\
    verticalPolarToCipicCoordinates
from home_platform.rendering import RgbRenderer
from home_platform.physics import Panda3dBulletPhysics_jz
TEST_DATA_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "data", "suncg")
modelpath= os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "data", "models")
class TestEvertAcoustics(unittest.TestCase):
    def testRenderHouseWithAcousticsPath(self):

        scene = SunCgSceneLoader.loadHouseFromJson(
            "0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)
        agentNp = scene.agents[0]
        model = Actor(os.path.join(modelpath,'eve'),  # Load our animated charachter
                         {'walk': os.path.join(modelpath,'eve_walk')})
#        model.setColor(LVector4f(np.random.uniform(), np.random.uniform(), np.random.uniform(), 1.0))
        model.setColor(LVector4f(0.75,0.70,0.8, 1.0))
        model.setTransform(TransformState.makeScale(0.15))
        eveNeck=model.controlJoint(None,'modelRoot','Neck')
        model.reparentTo(agentNp)
        # agentNp.setPos(LVecBase3f(45, -42, 1.6))
        # agentNp.seiiiitPos(LVecBase3f(40, -41.5, 1.6))
        agentNp.setPos(LVecBase3f(46, -42, 0.758729))
        agentNp.setHpr(-90, 0, 0)
        # physics = Panda3dBulletPhysics_jz(scene, TEST_SUNCG_DATA_DIR, objectMode='box',agentRadius=0.15, agentMode='sphere')

        # Define a sound source
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
        # objectNp.setPos(LVecBase3f(39, -40.5, 1.5))  # In the toilet
        objectNp.setPos(LVecBase3f(48, -42, 1.6))

        samplingRate = 16000.0
        # hrtf = CipicHRTF(os.path.join(TEST_DATA_DIR, 'hrtf',
        #
        #                               microtransform = [TransformState.makePos(LVecBase3f(0.15,0.03, 0.6)),TransformState.makePos(LVecBase3f(-0.15,0.03, 0.6))] ## two microphones on the ears'cipic_hrir.mat'), samplingRate) ## unused
        microtransform = [TransformState.makePos(LVecBase3f(0.15,0.03, 0.6)),TransformState.makePos(LVecBase3f(-0.15,0.03, 0.6)),TransformState.makePos(LVecBase3f(0,-0.15, 0.6))] ## third microphone on the tail
        #microtransform = [TransformState.makePos(LVecBase3f(0.15,0.03, 0.6)),TransformState.makePos(LVecBase3f(-0.15,0.03, 0.6))] ## two microphones on the ears

        acoustics = EvertAcoustics_jz(
            scene, None, samplingRate, maximumOrder=2, debug=True, microphoneTransform=microtransform,
        maxBufferLength=5)

        # Attach sound to object
        filename = os.path.join(TEST_DATA_DIR, 'audio', 'audio.wav')
        sound = EvertAudioSound(filename)
        acoustics.attachSoundToObject(sound, objectNp)
        sound.setLoop(True)
        sound.setLoopCount(7)
        sound.play()
        acoustics.step(0.1)

        physics = Panda3dBulletPhysics_jz(scene, TEST_SUNCG_DATA_DIR, objectMode='box',agentRadius=0.15, agentMode='sphere')

        renderer = RgbRenderer(scene, size=(128, 128), fov=70.0, cameraTransform=None)
        renderer.showRoomLayout(showCeilings=False, showWalls=True, showFloors=True)
        # Hide ceilings
        for nodePath in scene.scene.findAllMatches('**/layouts/*/physics/acoustics/*c'):
            nodePath.hide(BitMask32.allOn())
        viewer = Viewer_jz(scene,nbMicrophones=2, interactive=True,showPosition=False)
        # Configure the camera
        # NOTE: in Panda3D, the X axis points to the right, the Y axis is
        # forward, and Z is up
        center = agentNp.getNetTransform().getPos()
        transform = TransformState.makePosHpr(LVecBase3f(center.x, center.y-1, 15),
                                          LVecBase3f(0.0, -83.04, 0.0))
        viewer.cam.setTransform(transform)


        # impulse response
        imp = []
        jing = []
        jing.append(0)
        min_interval = 0.01
        Store_flag = True
        time_cur = 0
        time_gap = 0
        k = 0
        from scipy.io import wavfile
        # main loop
        clock = ClockObject.getGlobalClock()

        try:
            while True:
                # update physics
                dt = clock.getDt()
                print(dt)
                physics.step(dt)
                acoustics.step(dt)

                if k < 2:
                    k+=1
                    acoustics.getObservationsForAgent(agentNp.getName(),clearBuffer=True)

                else:
                    # time_cur += dt
                    # if time_cur <= 6:
                    #     time_gap += dt
                    #     if time_gap >= min_interval:
                    #         time_gap = 0
                    #         impulse = acoustics.calculateImpulseResponse(
                    #             objectNp.getName(), agentNp.getName())
                    #         imp.append(impulse.impulse[0])
                    #         jing.append(time_cur)
                    # elif Store_flag:
                    #     Store_flag = False
                    #     jing = jing[1:]
                    #     jing = np.asarray(jing)
                    #     imp.append(jing)
                    #     with open('Impulse.csv', 'w') as file:
                    #         mywriter = csv.writer(file, delimiter=',')
                    #         mywriter.writerows(imp)

                    acoustics.getObservationsForAgent(agentNp.getName(),clearBuffer=False)
                    time_cur += dt
                    if time_cur >= 5 and Store_flag:
                        Store_flag = False
                        imp= acoustics.calculateImpulseResponse(
                                 objectNp.getName(), agentNp.getName())
                        obs = acoustics.getObservationsForAgent(agentNp.getName())
                        data = np.array([obs['agent-0-mic0']['audio-buffer-0'],obs['agent-0-mic1']['audio-buffer-0']],
                                        dtype=np.float32).T
                        wavfile.write('Speech.wav', samplingRate, data)
                        ## plot of 2 microphones output
                        fig = plt.figure()
                        plt.plot(data[:,0],label='left-ears')
                        plt.plot(data[:,1],label='right-ears')
                        plt.xlabel('samples')
                        plt.ylabel('output signals')
                        plt.show(block=False)
                        plt.legend()
                        time.sleep(1.0)
                        plt.close(fig)

                        # plot of two microphones impulses
                        fig = plt.figure()
                        plt.plot(imp.impulse[0][0],label='left-ears')
                        plt.plot(imp.impulse[1][0],label='right-ears')
                        plt.xlabel('samples')
                        plt.ylabel('sound pressure(normalized)')
                        plt.show(block=False)
                        plt.legend()
                        time.sleep(20.0)
                        plt.close(fig)

                # Update viewer
                viewer.step()

                # file = open('Time.csv', 'w+')
                # with file:
                #     write_time = csv.writer(file)
                #     write_time.writerows(jing)
                # with open('Impulse.csv', 'w') as file:
                #     mywriter = csv.writer(file, delimiter=',')
                #     mywriter.writerows(imp)
                #     mywriter.writerows(jing)




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
