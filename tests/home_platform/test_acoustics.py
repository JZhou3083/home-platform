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

from panda3d.core import TransformState, LVecBase3f, LMatrix4f, BitMask32, AudioSound
from panda3d.core import LVector3f, TransformState, ClockObject, LVecBase3f, BitMask32, LVector4f,AudioSound
from direct.actor.Actor import Actor
from home_platform.suncg import SunCgSceneLoader, loadModel
from home_platform.core import Scene
from home_platform.utils import Viewer, vec3ToNumpyArray
from home_platform.acoustics import EvertAcoustics,EvertAcoustics_demon, CipicHRTF, FilterBank, \
    MaterialAbsorptionTable, AirAttenuationTable, EvertAudioSound, AudioPlayer,\
    interauralPolarToVerticalPolarCoordinates,\
    verticalPolarToInterauralPolarCoordinates,\
    verticalPolarToCipicCoordinates
from home_platform.rendering import RgbRenderer
from home_platform.physics import Panda3dBulletPhysics
TEST_DATA_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "data", "suncg")
modelpath= os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "data", "models")

class TestMaterialAbsorptionTable(unittest.TestCase):
    def testGetAbsorptionCoefficients(self):
        coefficientsDb, frequencies = MaterialAbsorptionTable.getAbsorptionCoefficients(category='hard surfaces',
                                                                                        material='marble floor')
        self.assertTrue(len(frequencies) == 7)
        self.assertTrue(len(coefficientsDb) == 7)
        self.assertTrue(np.all(coefficientsDb <= 0.0))

        coefficientsDb, frequencies = MaterialAbsorptionTable.getAbsorptionCoefficients(category='floor coverings',
                                                                                        material='carpet on hair felt or foam rubber')


class TestAirAttenuationTable(unittest.TestCase):
    def testGetAttenuations(self):

        for distance in [1.0, 24.0, 300.0]:
            for temperature in [10.0, 20.0, 35.0]:
                for relativeHumidity in [30.0, 55.0, 75.0]:
                    attenuationsDb, frequencies = AirAttenuationTable.getAttenuations(distance, temperature,
                                                                                      relativeHumidity)
                    self.assertTrue(len(frequencies) == 7)
                    self.assertTrue(len(attenuationsDb) == 7)
                    self.assertTrue(np.all(attenuationsDb <= 0.0))


class TestCipicHRTF(unittest.TestCase):
    def testInit(self):
        hrtf = CipicHRTF(filename=os.path.join(TEST_DATA_DIR, 'hrtf', 'cipic_hrir.mat'),
                         samplingRate=44100.0)
        self.assertTrue(np.array_equal(hrtf.impulses.shape, [25, 50, 2, 200]))

        hrtf = CipicHRTF(filename=os.path.join(TEST_DATA_DIR, 'hrtf', 'cipic_hrir.mat'),
                         samplingRate=16000.0)
        self.assertTrue(np.array_equal(hrtf.impulses.shape, [25, 50, 2, 72]))

    def testGetImpulseResponse(self):

        hrtf = CipicHRTF(filename=os.path.join(TEST_DATA_DIR, 'hrtf', 'cipic_hrir.mat'),
                         samplingRate=44100.0)

        for elevation, azimut in [(0.0, 0.0), (0.0, -90.0), (0.0, 90.0), (-120.0, 0.0)]:

            impulse = hrtf.getImpulseResponse(azimut, elevation)
            self.assertTrue(np.array_equal(impulse.shape, [2, 200]))

            fig = plt.figure()
            plt.title('azimut = %f, elevation = %f' % (azimut, elevation))
            plt.plot(impulse[0], color='b', label='Left channel')
            plt.plot(impulse[1], color='g', label='Right channel')
            plt.legend()
            plt.show(block=False)
            time.sleep(1.0)
            plt.close(fig)

    def testResample(self):
        hrtf = CipicHRTF(filename=os.path.join(TEST_DATA_DIR, 'hrtf', 'cipic_hrir.mat'),
                         samplingRate=44100.0)
        hrtf.resample(newSamplingRate=16000.0)


class TestFilterBank(unittest.TestCase):
    def testInit(self):
        n = 256
        centerFrequencies = np.array(
            [125, 500, 1000, 2000, 4000], dtype=np.float)
        samplingRate = 16000
        filterbank = FilterBank(n, centerFrequencies, samplingRate)
        impulse = filterbank.getScaledImpulseResponse()
        self.assertTrue(impulse.ndim == 1)
        self.assertTrue(impulse.shape[0] == n + 1)

        n = 511
        centerFrequencies = np.array(
            [125, 500, 1000, 2000, 4000], dtype=np.float)
        samplingRate = 16000
        filterbank = FilterBank(n, centerFrequencies, samplingRate)
        impulse = filterbank.getScaledImpulseResponse()
        self.assertTrue(impulse.ndim == 1)
        self.assertTrue(impulse.shape[0] == n)

    def testDisplay(self):
        n = 257
        centerFrequencies = np.array(
            [125, 500, 1000, 2000, 4000], dtype=np.float)
        samplingRate = 16000
        filterbank = FilterBank(n, centerFrequencies, samplingRate)
        fig = filterbank.display(merged=False)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        fig = filterbank.display(merged=True)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)

        n = 257
        scales = np.array([1.0, 0.5, 0.25, 0.5, 0.05])
        centerFrequencies = np.array(
            [125, 500, 1000, 2000, 4000], dtype=np.float)
        samplingRate = 16000
        filterbank = FilterBank(n, centerFrequencies, samplingRate)
        fig = filterbank.display(scales, merged=False)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        fig = filterbank.display(scales, merged=True)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)

        impulse = filterbank.getScaledImpulseResponse(scales)
        fig = plt.figure()
        plt.plot(impulse)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)


class TestEvertAcoustics(unittest.TestCase):
    def testInit(self):

        samplingRate = 16000.0
        scene = SunCgSceneLoader.loadHouseFromJson(
            "0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)
        hrtf = CipicHRTF(os.path.join(TEST_DATA_DIR, 'hrtf',
                                      'cipic_hrir.mat'), samplingRate)

        engine = EvertAcoustics(scene, hrtf, samplingRate,
                                maximumOrder=2, debug=True)
        engine.destroy()

    def testRenderSimpleCubeRoom(self):

        samplingRate = 16000.0
        scene = Scene()
        hrtf = CipicHRTF(os.path.join(TEST_DATA_DIR, 'hrtf',
                                      'cipic_hrir.mat'), samplingRate)

        viewer = Viewer(scene, interactive=False)

        # Define a simple cube (10 x 10 x 10 m) as room geometry
        roomSize = 10.0
        modelId = 'room-0'
        modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'cube.egg')
        layoutNp = scene.scene.attachNewNode('layouts')
        objectNp = layoutNp.attachNewNode('object-' + modelId)
        objectNp.setTag('acoustics-mode', 'obstacle')
        model = loadModel(modelFilename)
        model.setName('model-' + modelId)
        model.setTransform(TransformState.makeScale(roomSize))
        model.setRenderModeWireframe()
        model.reparentTo(objectNp)
        objectNp.setPos(LVecBase3f(0.0, 0.0, 0.0))

        # Define a sound source
        sourceSize = 0.25
        modelId = 'source-0'
        modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'sphere.egg')
        objectsNp = scene.scene.attachNewNode('objects')
        objectNp = objectsNp.attachNewNode('object-' + modelId)
        objectNp.setTag('acoustics-mode', 'source')
        model = loadModel(modelFilename)
        model.setName('model-' + modelId)
        model.setTransform(TransformState.makeScale(sourceSize))
        model.setColor(LVector4f(0,0,0, 1.0))
        model.reparentTo(objectNp)
        objectNp.setPos(LVecBase3f(0.0, 0.0, 0.0))

        acoustics = EvertAcoustics(scene, hrtf, samplingRate, maximumOrder=3, materialAbsorption=False,
                                   frequencyDependent=False, debug=True)

        # Attach sound to object
        filename = os.path.join(TEST_DATA_DIR, 'audio', 'toilet.ogg')
        sound = EvertAudioSound(filename)
        acoustics.attachSoundToObject(sound, objectNp)

        acoustics.step(0.1)
        center = acoustics.world.getCenter()
        self.assertTrue(np.allclose(
            acoustics.world.getMaxLength() / 1000.0, roomSize))
        self.assertTrue(np.allclose(
            [center.x, center.y, center.z], [0.0, 0.0, 0.0]))
        self.assertTrue(acoustics.world.numElements() == 12)
        self.assertTrue(acoustics.world.numConvexElements() == 12)

        # Configure the camera
        # NOTE: in Panda3D, the X axis points to the right, the Y axis is
        # forward, and Z is up
        mat = np.array([0.999992, 0.00394238, 0, 0,
                        -0.00295702, 0.750104, -0.661314, 0,
                        -0.00260737, 0.661308, 0.75011, 0,
                        0.0, -25.0, 22, 1])
        mat = LMatrix4f(*mat.ravel())
        viewer.cam.setMat(mat)

        agentNp = scene.agents[0]
        agentNp.setPos(LVecBase3f(0.25 * roomSize, -
                                  0.25 * roomSize, 0.3 * roomSize))
        for _ in range(10):
            viewer.step()
        time.sleep(1.0)

        agentNp.setPos(LVecBase3f(0.35 * roomSize, -
                                  0.35 * roomSize, 0.4 * roomSize))
        for _ in range(10):
            viewer.step()
        time.sleep(1.0)

        agentNp.setPos(LVecBase3f(-0.25 * roomSize,
                                  0.25 * roomSize, -0.3 * roomSize))
        for _ in range(10):
            viewer.step()
        time.sleep(20.0)

        # Calculate and show impulse responses
        impulse = acoustics.calculateImpulseResponse(
            objectNp.getName(), agentNp.getName())

        fig = plt.figure()
        plt.plot(impulse.impulse[0], color='b', label='Left channel')
        plt.plot(impulse.impulse[1], color='g', label='Right channel')
        plt.legend()
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)

        acoustics.destroy()
        viewer.destroy()
        viewer.graphicsEngine.removeAllWindows()

    def testRenderHouse(self):

        scene = SunCgSceneLoader.loadHouseFromJson(
            "0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)

        samplingRate = 16000.0
        hrtf = CipicHRTF(os.path.join(TEST_DATA_DIR, 'hrtf',
                                      'cipic_hrir.mat'), samplingRate)
        acoustics = EvertAcoustics(
            scene, hrtf, samplingRate, maximumOrder=2, debug=True)


        # cameraTransform = TransformState.makePosHpr(LVector3f(0.0,  -0.3, -3),LVector3f(0, 180, 0))
        # renderer = RgbRenderer(scene, size=(128, 128), fov=70.0, cameraTransform=cameraTransform)
        # renderer.showRoomLayout(showCeilings=False, showWalls=True, showFloors=True)
        acoustics.step(0.0)


        # Hide ceilings
        for nodePath in scene.scene.findAllMatches('**/layouts/*/acoustics/*c'):
            nodePath.hide(BitMask32.allOn())

        viewer = Viewer(scene, interactive=False)

        # Configure the camera
        # NOTE: in Panda3D, the X axis points to the right, the Y axis is
        # forward, and Z is up
        mat = np.array([0.999992, 0.00394238, 0, 0,
                        -0.00295702, 0.750104, -0.661314, 0,
                        -0.00260737, 0.661308, 0.75011, 0,
                        43.621, -55.7499, 12.9722, 1])
        mat = LMatrix4f(*mat.ravel())
        viewer.cam.setMat(mat)

        for _ in range(20):
            acoustics.step(dt=0.1)
            viewer.step()
        time.sleep(1.0)

        acoustics.destroy()
        viewer.destroy()
        # renderer.destroy()
        viewer.graphicsEngine.removeAllWindows()

    def testRenderHouseWithAcousticsPath(self):

        scene = SunCgSceneLoader.loadHouseFromJson(
            "0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)
        agentNp = scene.agents[0]
        model = Actor(os.path.join(modelpath,'eve'),  # Load our animated charachter
                         {'walk': os.path.join(modelpath,'eve_walk')})
#        model.setColor(LVector4f(np.random.uniform(), np.random.uniform(), np.random.uniform(), 1.0))
        model.setColor(LVector4f(0.75,0.70,0.8, 1.0))
        model.setTransform(TransformState.makeScale(0.15))
        model.reparentTo(agentNp)
        agentNp.setPos(LVecBase3f(45, -44, 1.6))
        # agentNp.setPos(LVecBase3f(40, -41.5, 1.6))
        # agentNp.setPos(LVecBase3f(42.7891, -39.904, 0.758729))
        agentNp.setHpr(0, 0, 0)
        physics = Panda3dBulletPhysics(scene, TEST_SUNCG_DATA_DIR, objectMode='box',
                                    agentRadius=0.15, agentMode='sphere')

        # Define a sound source
        sourceSize = 0.25
        modelId = 'source-0'
        modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'sphere.egg')
        objectsNp = scene.scene.attachNewNode('objects')
        objectsNp.setTag('acoustics-mode', 'source')
        objectNp = objectsNp.attachNewNode('object-' + modelId)
        model = loadModel(modelFilename)
        model.setName('model-' + modelId)
        model.setTransform(TransformState.makeScale(sourceSize))
        model.reparentTo(objectNp)
        objectNp.setPos(LVecBase3f(39, -40.5, 1.5))



        samplingRate = 16000.0
        hrtf = CipicHRTF(os.path.join(TEST_DATA_DIR, 'hrtf',
                                      'cipic_hrir.mat'), samplingRate)
        microtransform = TransformState.makePos(LVecBase3f(0.15,0, 0),)
        acoustics = EvertAcoustics(
            scene, hrtf, samplingRate, maximumOrder=2, debug=True,microphoneTransform=microtransform)
        # Attach sound to object
        filename = os.path.join(TEST_DATA_DIR, 'audio', 'radio.ogg')
        sound = EvertAudioSound(filename)
        acoustics.attachSoundToObject(sound, objectNp)
        sound.play()

        cameraTransform = TransformState.makePosHpr(LVector3f(0.0,  -0.3, 0),LVector3f(0, 180, 0))
        renderer = RgbRenderer(scene, size=(128, 128), fov=70.0, cameraTransform=cameraTransform)
        renderer.showRoomLayout(showCeilings=False, showWalls=True, showFloors=True)

        acoustics.step(0.0)
        # Hide ceilings
        for nodePath in scene.scene.findAllMatches('**/layouts/*/physics/acoustics/*c'):
            nodePath.hide(BitMask32.allOn())
        viewer = Viewer(scene, interactive=False)

        # Configure the camera
        # NOTE: in Panda3D, the X axis points to the right, the Y axis is
        # forward, and Z is up
        center = agentNp.getNetTransform().getPos()
        mat = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [center.x, center.y, 20, 1]])
        mat = LMatrix4f(*mat.ravel())
        viewer.cam.setMat(mat)
        # main loop
        clock = ClockObject.getGlobalClock()
        try:
            while True:

                # update physics
                dt = clock.getDt()


                # agentRbNp = agentNp.find('**/+BulletRigidBodyNode')
                agentNp.setY(agentNp.getY()+0.05*dt)
                # curPos=vec3ToNumpyArray(agentRbNp.getNetTransform().getPos())
                # agentRbNp.setY(agentRbNp.getY()+50*dt)
                physics.step(dt)

                # Update viewer
                viewer.step()
        except KeyboardInterrupt:
            pass

        viewer.destroy()
        physics.destroy()
        viewer.graphicsEngine.removeAllWindows()

        # # Calculate and show impulse responses
        # impulse = acoustics.calculateImpulseResponse(
        #     objectNp.getName(), agentNp.getName())
        #
        # fig = plt.figure()
        # plt.plot(impulse.impulse[0], color='b', label='Left channel')
        # plt.plot(impulse.impulse[1], color='g', label='Right channel')
        # plt.legend()
        # plt.show(block=False)
        # time.sleep(10.0)
        # plt.close(fig)
        renderer.destroy()
        acoustics.destroy()

    def testAttachSoundToObject(self):

        samplingRate = 16000.0
        scene = SunCgSceneLoader.loadHouseFromJson(
            "0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)
        acoustics = EvertAcoustics(
            scene, samplingRate=samplingRate, maximumOrder=2)

        filename = os.path.join(TEST_DATA_DIR, 'audio', 'toilet.ogg')
        sound = EvertAudioSound(filename)

        objNode = scene.scene.find('**/object-83*')
        acoustics.attachSoundToObject(sound, objNode)

    def testStep(self):

        scene = SunCgSceneLoader.loadHouseFromJson(
            "0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)

        agentNp = scene.agents[0]
        agentNp.setPos(LVecBase3f(45, -42.5, 1.6))
        agentNp.setHpr(45, 0, 0)

        # Define a sound source
        sourceSize = 0.25
        modelId = 'source-0'
        modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'sphere.egg')
        objectsNp = scene.scene.attachNewNode('objects')
        objectsNp.setTag('acoustics-mode', 'source')
        objectNp = objectsNp.attachNewNode('object-' + modelId)
        model = loadModel(modelFilename)
        model.setName('model-' + modelId)
        model.setTransform(TransformState.makeScale(sourceSize))
        model.reparentTo(objectNp)
        objectNp.setPos(LVecBase3f(39, -40.5, 1.5))

        samplingRate = 16000.0
        hrtf = CipicHRTF(os.path.join(TEST_DATA_DIR, 'hrtf',
                                      'cipic_hrir.mat'), samplingRate)
        acoustics = EvertAcoustics_demon(
            scene, hrtf, samplingRate, maximumOrder=2, maxBufferLength=30.0)

        # Attach sound to object
        filename = os.path.join(TEST_DATA_DIR, 'audio', 'toilet.ogg')
        sound = EvertAudioSound(filename)
        acoustics.attachSoundToObject(sound, objectNp)
        sound.setLoop(True)
        sound.setLoopCount(1)
        sound.play()

        for i, dt in enumerate([5.0, 20.0, 10.0]):

            acoustics.step(dt)
            if i == 0:
                self.assertTrue(sound.status() == AudioSound.PLAYING)
            elif i > 1:
                self.assertTrue(sound.status() == AudioSound.READY)
            inbuf = acoustics.srcBuffers[sound]
            outbuf = acoustics.outBuffers[agentNp.getName()]

            fig = plt.figure()
            plt.subplot(121)
            plt.plot(inbuf)
            plt.subplot(122)
            plt.plot(outbuf.T)
            plt.show(block=False)
            time.sleep(4.0)
            plt.close(fig)

    def testMultipleSources(self):

        scene = SunCgSceneLoader.loadHouseFromJson(
            "0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)

        agentNp = scene.agents[0]
        agentNp.setPos(LVecBase3f(45, -42.5, 1.6))
        agentNp.setHpr(45, 0, 0)

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
        acoustics = EvertAcoustics(scene, hrtf, samplingRate, maximumOrder=2)

        audioFilenames = ['toilet.ogg', 'radio.ogg']
        for audioFilename, source in zip(audioFilenames, sources):
            # Attach sound to object
            filename = os.path.join(TEST_DATA_DIR, 'audio', audioFilename)
            sound = EvertAudioSound(filename)
            acoustics.attachSoundToObject(sound, source)
            sound.setLoop(True)
            sound.setLoopCount(1)
            sound.play()

        for _ in range(20):
            acoustics.step(dt=0.1)
            obs = acoustics.getObservationsForAgent(agentNp.getName())
            self.assertTrue('audio-buffer-right' in obs)
            self.assertTrue('audio-buffer-left' in obs)
            self.assertTrue(np.array_equal(
                obs['audio-buffer-right'].shape, obs['audio-buffer-left'].shape))

    def testListen(self):

        scene = SunCgSceneLoader.loadHouseFromJson(
            "0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)

        agentNp = scene.agents[0]
        agentNp.setPos(LVecBase3f(45, -42.5, 1.6))
        agentNp.setHpr(45, 0, 0)

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
        acoustics = EvertAcoustics_demon(
            scene, hrtf, samplingRate, maximumOrder=2, maxBufferLength=30.0)

        # audioFilenames = ['toilet.ogg', 'radio.ogg']
        audioFilenames = ['audio.wav','radio.ogg']
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

        obs = acoustics.getObservationsForAgent(agentNp.getName())
        data = np.array([obs['audio-buffer-left'],
                         obs['audio-buffer-right']], dtype=np.float32).T
        self.assertTrue(np.allclose(
            data.shape[0] / samplingRate, 15.0, atol=1e-3))

        # from scipy.io import wavfile
        # wavfile.write('output.wav', samplingRate, data)

        fig = plt.figure()
        plt.plot(data)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)

    def testAddAmbientSound(self):

        scene = SunCgSceneLoader.loadHouseFromJson(
            "0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)

        agentNp = scene.agents[0]
        # agentNp.setPos(LVecBase3f(45, -42.5, 1.6))
        agentNp.setPos(LVecBase3f(39, -40.5, 1.5))

        agentNp.setHpr(45, 0, 0)

        # Define a sound source
        sourceSize = 0.25
        modelId = 'source-0'
        modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'sphere.egg')
        objectsNp = scene.scene.attachNewNode('objects')
        objectsNp.setTag('acoustics-mode', 'source')
        objectNp = objectsNp.attachNewNode('object-' + modelId)
        model = loadModel(modelFilename)
        model.setName('model-' + modelId)
        model.setTransform(TransformState.makeScale(sourceSize))
        model.reparentTo(objectNp)
        objectNp.setPos(LVecBase3f(39, -40.5, 1.5))

        samplingRate = 16000.0
        hrtf = CipicHRTF(os.path.join(TEST_DATA_DIR, 'hrtf',
                                      'cipic_hrir.mat'), samplingRate)
        acoustics = EvertAcoustics(
            scene, None, samplingRate, maximumOrder=2, maxBufferLength=30.0)

        # Attach sound to object
        filename = os.path.join(TEST_DATA_DIR, 'audio', 'audio.wav')
        sound = EvertAudioSound(filename)
        acoustics.attachSoundToObject(sound, objectNp)
        sound.setLoop(True)
        sound.setLoopCount(1)

        # # Add ambient sound
        # filename = os.path.join(TEST_DATA_DIR, 'audio', 'radio.ogg')
        # ambientSound = EvertAudioSound(filename)
        # ambientSound.setLoop(True)
        # ambientSound.setLoopCount(0)
        # ambientSound.setVolume(0.25)
        # acoustics.addAmbientSound(ambientSound)

        # ambientSound.play()
        # acoustics.step(dt=5.0)
        sound.play()
        acoustics.step(dt=7.0)

        obs = acoustics.getObservationsForAgent(agentNp.getName())

        data = np.array(obs['audio-buffer-0'],
                         dtype=np.float32).T
        self.assertTrue(np.allclose(
            data.shape[0] / samplingRate, 7.0, atol=1e-3))

        from scipy.io import wavfile
        wavfile.write('Speech.wav', samplingRate, data)

        fig = plt.figure()
        plt.plot(data)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)


class TestEvertAudioSound(unittest.TestCase):

    def testInit(self):

        filename = os.path.join(TEST_DATA_DIR, 'audio', 'toilet.ogg')
        sound = EvertAudioSound(filename)
        self.assertTrue(sound.data.ndim == 1)
        self.assertTrue(sound.data.dtype == np.float)
        self.assertTrue(sound.samplingRate == 16000.0)
        self.assertTrue(np.allclose(sound.length(), 15.846, atol=1e-2))

        sound.resample(8000.0)
        self.assertTrue(sound.data.ndim == 1)
        self.assertTrue(sound.data.dtype == np.float)
        self.assertTrue(np.allclose(sound.length(), 15.846, atol=1e-2))


class TestAudioPlayer(unittest.TestCase):

    def testUpdate(self):

        scene = SunCgSceneLoader.loadHouseFromJson(
            "0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)

        agentNp = scene.agents[0]
        agentNp.setPos(LVecBase3f(45, -42.5, 1.6))
        agentNp.setHpr(45, 0, 0)

        samplingRate = 16000.0
        hrtf = CipicHRTF(os.path.join(TEST_DATA_DIR, 'hrtf',
                                      'cipic_hrir.mat'), samplingRate)
        acoustics = EvertAcoustics(
            scene, hrtf, samplingRate, maximumOrder=2, maxBufferLength=30.0)

        # Add ambient sound
        filename = os.path.join(TEST_DATA_DIR, 'audio', 'radio.ogg')
        ambientSound = EvertAudioSound(filename)
        ambientSound.setLoop(True)
        ambientSound.setLoopCount(0)
        acoustics.addAmbientSound(ambientSound)
        ambientSound.play()

        acoustics.step(0.0)

        from direct.task.TaskManagerGlobal import taskMgr
        AudioPlayer(acoustics)
        for _ in range(10):
            taskMgr.step()


class TestFunctions(unittest.TestCase):

    def testInterauralPolarToVerticalPolarCoordinates(self):
        elevations = np.array([0.0, 90.0, 90.0, 90.0, 90.0])
        azimuts = np.array([0.0, 0.0, 90.0, -90.0, 45.0])

        vertElevations, vertAzimuts = interauralPolarToVerticalPolarCoordinates(
            elevations, azimuts)
        self.assertTrue(np.allclose(vertElevations, [
                        0.0, 90.0, 0.0, 0.0, 45.0], atol=1e-6))
        self.assertTrue(np.allclose(
            vertAzimuts, [0.0, 0.0, 90.0, -90.0, 90.0], atol=1e-6))

    def testVerticalPolarToInterauralPolarCoordinates(self):
        elevations = np.array([0.0, 90.0, 0.0, 0.0, 45.0])
        azimuts = np.array([0.0, 0.0, 90.0, -90.0, 90.0])

        vertElevations, vertAzimuts = verticalPolarToInterauralPolarCoordinates(
            elevations, azimuts)
        self.assertTrue(np.allclose(vertElevations, [
                        0.0, 90.0, 0.0, 0.0, 90.0], atol=1e-6))
        self.assertTrue(np.allclose(
            vertAzimuts, [0.0, 0.0, 90.0, -90.0, 45.0], atol=1e-6))

    def testVerticalPolarToCipicCoordinates(self):
        elevations = np.array([0.0, 90.0, 0.0, 0.0, 45.0, -180.0])
        azimuts = np.array([0.0, 0.0, 90.0, -90.0, 90.0, 0.0])

        cipicElevations, cipicAzimuts = verticalPolarToCipicCoordinates(
            elevations, azimuts)
        self.assertTrue(np.allclose(cipicElevations, [
                        0.0, 90.0, 0.0, 0.0, 90.0, 180], atol=1e-6))
        self.assertTrue(np.allclose(
            cipicAzimuts, [0.0, 0.0, 90.0, -90.0, 45.0, 0.0], atol=1e-6))


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
