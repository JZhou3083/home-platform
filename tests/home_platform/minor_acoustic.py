import os
import time
import logging
import numpy as np
import unittest
import matplotlib.pyplot as plt

from panda3d.core import TransformState, LVecBase3f, LMatrix4f, BitMask32, AudioSound

from home_platform.suncg import SunCgSceneLoader, loadModel
from home_platform.core import Scene
from home_platform.utils import Viewer
from home_platform.acoustics import EvertAcoustics, CipicHRTF, FilterBank, \
    MaterialAbsorptionTable, AirAttenuationTable, EvertAudioSound, AudioPlayer,\
    interauralPolarToVerticalPolarCoordinates,\
    verticalPolarToInterauralPolarCoordinates,\
    verticalPolarToCipicCoordinates

TEST_DATA_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "data", "suncg")

class TestEvertAcoustics(unittest.TestCase):
    def testRenderHouseWithAcousticsPath(self):

        scene = SunCgSceneLoader.loadHouseFromJson(
            "0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)

        agentNp = scene.agents[0]
#        agentNp.setPos(LVecBase3f(45, -42.5, 1.6))
        agentNp.setPos(LVecBase3f(39, -40.5, 1.6))
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
            scene, hrtf, samplingRate, maximumOrder=2, debug=True)

        # Attach sound to object
        filename = os.path.join(TEST_DATA_DIR, 'audio', 'toilet.ogg')
        sound = EvertAudioSound(filename)
        acoustics.attachSoundToObject(sound, objectNp)
        sound.play()

        acoustics.step(0.0)

#        # Hide ceilings
#        for nodePath in scene.scene.findAllMatches('**/layouts/*/acoustics/*c'):
#            nodePath.hide(BitMask32.allOn())

#        viewer = Viewer(scene, interactive=False)

##        # Configure the camera
#        # NOTE: in Panda3D, the X axis points to the right, the Y axis is
#        # forward, and Z is up
#        center = agentNp.getNetTransform().getPos()
#        mat = np.array([[1.0, 0.0, 0.0, 0.0],
#                        [0.0, 0.0, -1.0, 0.0],
#                        [0.0, 1.0, 0.0, 0.0],
#                        [center.x, center.y, 20, 1]])
#        mat = LMatrix4f(*mat.ravel())
#        print(mat)
#        viewer.cam.setMat(mat)

#        for _ in range(20):
#            viewer.step()
#        time.sleep(15.0)

#        viewer.destroy()
#        viewer.graphicsEngine.removeAllWindows()

        # Calculate and show impulse responses
        impulse = acoustics.calculateImpulseResponse(
            objectNp.getName(), agentNp.getName())

        fig = plt.figure()
        plt.plot(impulse.impulse[0], color='b', label='Left channel')
        plt.plot(impulse.impulse[1], color='g', label='Right channel')
        plt.legend()
        plt.show(block=False)
        time.sleep(5.0)
        plt.close(fig)

        acoustics.destroy()
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
