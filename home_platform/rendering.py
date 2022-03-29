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

from __future__ import print_function

import os
import sys
import six
import logging
import numpy as np
import imageio

from panda3d.core import VBase4, PointLight, AmbientLight, AntialiasAttrib, \
    GeomVertexReader, GeomTristrips, GeomTriangles, LineStream, SceneGraphAnalyzer, \
    LVecBase3f, LVecBase4f, TransparencyAttrib, ColorAttrib, TextureAttrib, GeomEnums, \
    BitMask32, RenderState, LColor, LVector4f

from panda3d.core import GraphicsEngine, GraphicsPipeSelection, Loader, RescaleNormalAttrib, \
    Texture, GraphicsPipe, GraphicsOutput, FrameBufferProperties, WindowProperties, Camera, PerspectiveLens, ModelNode

from home_platform.core import World
from home_platform.suncg import ModelCategoryMapping
from home_platform.constants import MODEL_CATEGORY_COLOR_MAPPING

logger = logging.getLogger(__name__)

MODEL_DATA_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "data", "models")


class Panda3dRenderer(World):
    def __init__(self, scene, size=(512, 512), shadowing=False, mode='offscreen', zNear=0.1, zFar=1000.0, fov=40.0,
                 depth=True, modelLightsInfo=None, cameraTransform=None):

        super(Panda3dRenderer, self).__init__()

        self.__dict__.update(scene=scene, size=size, mode=mode, zNear=zNear, zFar=zFar, fov=fov,
                             depth=depth, shadowing=shadowing, modelLightsInfo=modelLightsInfo,
                             cameraTransform=cameraTransform)

        self.cameraMask = BitMask32.bit(0)
        self.graphicsEngine = GraphicsEngine.getGlobalPtr()
        self.loader = Loader.getGlobalPtr()
        self.graphicsEngine.setDefaultLoader(self.loader)

        # Change some scene attributes for rendering
        self.scene.scene.setAttrib(RescaleNormalAttrib.makeDefault())
        self.scene.scene.setTwoSided(True)

        self._initModels()

        selection = GraphicsPipeSelection.getGlobalPtr()
        self.pipe = selection.makeDefaultPipe()
        logger.debug('Using %s' % (self.pipe.getInterfaceName()))

        # Attach a camera to every agent in the scene
        self.cameras = []
        for agentNp in self.scene.scene.findAllMatches('**/agents/agent*'):
            camera = agentNp.attachNewNode(ModelNode('camera-rgbd'))
#        for neck in self.scene.scene.findAllMatches('**/agents/agent*/model*/Neck'):

#            camera = neck.attachNewNode(ModelNode('camera-rgbd'))

            if self.cameraTransform is not None:
                camera.setTransform(cameraTransform)
            camera.node().setPreserveTransform(ModelNode.PTLocal)
            self.cameras.append(camera)

        self.rgbBuffers = dict()
        self.rgbTextures = dict()
        self.depthBuffers = dict()
        self.depthTextures = dict()

        self._initRgbCapture()
        if self.depth:
            self._initDepthCapture()

        self._addDefaultLighting()

        self.scene.worlds['render'] = self

    def _initModels(self):

        for model in self.scene.scene.findAllMatches('**/+ModelNode'):

            objectNp = model.getParent()
            rendererNp = objectNp.attachNewNode('render')
            model = model.copyTo(rendererNp)

            # Set the model to be visible only to this camera
            model.hide(BitMask32.allOn())
            model.show(self.cameraMask)

            # Reparent render node below the existing physic node (if any)
            physicsNp = objectNp.find('**/physics')
            if not physicsNp.isEmpty():
                rendererNp.reparentTo(physicsNp)

    def _initRgbCapture(self):

        for camera in self.cameras:

            camNode = Camera('RGB camera')
            camNode.setCameraMask(self.cameraMask)
            lens = PerspectiveLens()
            lens.setFov(self.fov)
            lens.setAspectRatio(float(self.size[0]) / float(self.size[1]))
            lens.setNear(self.zNear)
            lens.setFar(self.zFar)
            camNode.setLens(lens)
            camNode.setScene(self.scene.scene)
            cam = camera.attachNewNode(camNode)

            winprops = WindowProperties.size(self.size[0], self.size[1])
            fbprops = FrameBufferProperties.getDefault()
            fbprops = FrameBufferProperties(fbprops)
            fbprops.setRgbaBits(8, 8, 8, 0)

            flags = GraphicsPipe.BFFbPropsOptional
            if self.mode == 'onscreen':
                flags = flags | GraphicsPipe.BFRequireWindow
            elif self.mode == 'offscreen':
                flags = flags | GraphicsPipe.BFRefuseWindow
            else:
                raise Exception('Unsupported rendering mode: %s' % (self.mode))

            buf = self.graphicsEngine.makeOutput(self.pipe, 'RGB-buffer-Rendering', 0, fbprops,
                                                 winprops, flags)
            if buf is None:
                raise Exception('Unable to create RGB buffer')

            # Set to render at the end
            buf.setSort(10000)

            dr = buf.makeDisplayRegion()
            dr.setSort(0)
            dr.setCamera(cam)
            dr = camNode.getDisplayRegion(0)

            tex = Texture()
            tex.setFormat(Texture.FRgb8)
            tex.setComponentType(Texture.TUnsignedByte)
            buf.addRenderTexture(
                tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)
            # XXX: should use tex.setMatchFramebufferFormat(True)?

            agent = camera.getParent()
            self.rgbBuffers[agent.getName()] = buf
            self.rgbTextures[agent.getName()] = tex

    def _initDepthCapture(self):

        for camera in self.cameras:

            camNode = Camera('Depth camera')
            camNode.setCameraMask(self.cameraMask)
            lens = PerspectiveLens()
            lens.setFov(self.fov)
            lens.setAspectRatio(float(self.size[0]) / float(self.size[1]))
            lens.setNear(self.zNear)
            lens.setFar(self.zFar)
            camNode.setLens(lens)
            camNode.setScene(self.scene.scene)
            cam = camera.attachNewNode(camNode)

            winprops = WindowProperties.size(self.size[0], self.size[1])
            fbprops = FrameBufferProperties.getDefault()
            fbprops = FrameBufferProperties(fbprops)
            fbprops.setRgbColor(False)
            fbprops.setRgbaBits(0, 0, 0, 0)
            fbprops.setStencilBits(0)
            fbprops.setMultisamples(0)
            fbprops.setBackBuffers(0)
            fbprops.setDepthBits(16)

            flags = GraphicsPipe.BFFbPropsOptional
            if self.mode == 'onscreen':
                flags = flags | GraphicsPipe.BFRequireWindow
            elif self.mode == 'offscreen':
                flags = flags | GraphicsPipe.BFRefuseWindow
            else:
                raise Exception('Unsupported rendering mode: %s' % (self.mode))

            buf = self.graphicsEngine.makeOutput(self.pipe, 'Depth buffer', 0, fbprops,
                                                 winprops, flags)
            if buf is None:
                raise Exception('Unable to create depth buffer')

            # Set to render at the end
            buf.setSort(10000)

            dr = buf.makeDisplayRegion()
            dr.setSort(0)
            dr.setCamera(cam)
            dr = camNode.getDisplayRegion(0)

            tex = Texture()
            tex.setFormat(Texture.FDepthComponent)
            tex.setComponentType(Texture.TFloat)
            buf.addRenderTexture(
                tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)
            # XXX: should use tex.setMatchFramebufferFormat(True)?

            agent = camera.getParent()
            self.depthBuffers[agent.getName()] = buf
            self.depthTextures[agent.getName()] = tex

    def setWireframeOnly(self):
        self.scene.scene.setRenderModeWireframe()

    def showRoomLayout(self, showCeilings=True, showWalls=True, showFloors=True):

        for np in self.scene.scene.findAllMatches('**/layouts/**/render/*c'):
            if showCeilings:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

        for np in self.scene.scene.findAllMatches('**/layouts/**/render/*w'):
            if showWalls:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

        for np in self.scene.scene.findAllMatches('**/layouts/**/render/*f'):
            if showFloors:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

    def destroy(self):
        self.graphicsEngine.removeAllWindows()
        del self.pipe

    def getRgbImages(self, channelOrder="RGB"):
        images = dict()
        for name, tex in six.iteritems(self.rgbTextures):

            # XXX: not sure about calling makeRamImage() before getting the image data, since it returns an empty image
            # and overwrite any previously rendered image. We may just call it
            # once when we create the texture.
            if not tex.mightHaveRamImage():
                tex.makeRamImage()

            if sys.version_info[0] < 3:
                data = tex.getRamImageAs(channelOrder).getData()   # Python 2
            else:
                # NOTE: see https://github.com/panda3d/panda3d/issues/173
                data = bytes(memoryview(
                    tex.getRamImageAs(channelOrder)))  # Python 3

            # Must match Texture.TUnsignedByte
            image = np.frombuffer(data, dtype=np.uint8)

            image.shape = (tex.getYSize(), tex.getXSize(), len(channelOrder))
            image = np.flipud(image)
            images[name] = image

        return images

    def getDepthImages(self, mode='normalized'):

        images = dict()
        if self.depth:

            for name, tex in six.iteritems(self.depthTextures):
                # XXX: not sure about calling makeRamImage() before getting the image data, since it returns an empty image
                # and overwrite any previously rendered image. We may just call
                # it once when we create the texture.
                if not tex.mightHaveRamImage():
                    tex.makeRamImage()

                if sys.version_info[0] < 3:
                    data = tex.getRamImage().getData()   # Python 2
                else:
                    # NOTE: see https://github.com/panda3d/panda3d/issues/173
                    data = bytes(memoryview(tex.getRamImage()))  # Python 3

                nbBytesComponentFromData = len(
                    data) / (tex.getYSize() * tex.getXSize())
                if nbBytesComponentFromData == 4:
                    # Must match Texture.TFloat
                    depthImage = np.frombuffer(data, dtype=np.float32)

                elif nbBytesComponentFromData == 2:
                    # NOTE: This can happen on some graphic hardware, where unsigned 16-bit data is stored
                    # despite setting the texture component type to 32-bit
                    # floating point.
                    # Must match Texture.TFloat
                    depthImage = np.frombuffer(data, dtype=np.uint16)
                    depthImage = depthImage.astype(np.float32) / 65535

                depthImage.shape = (tex.getYSize(), tex.getXSize())
                depthImage = np.flipud(depthImage)

                if mode == 'distance':
                    # NOTE: in Panda3d, the returned depth image seems to be
                    # already linearized
                    depthImage = self.zNear + depthImage / \
                        (self.zFar - self.zNear)

                    # Adapted from: https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
                    # depthImage = 2.0 * depthImage - 1.0
                    # depthImage = 2.0 * self.zNear * self.zFar / (self.zFar + self.zNear - depthImage * (self.zFar - self.zNear))

                elif mode == 'normalized':
                    # Nothing to do
                    pass
                else:
                    raise Exception(
                        'Unsupported output depth image mode: %s' % (mode))

                images[name] = depthImage
        else:

            for name in six.iterkeys(self.depthTextures):
                images[name] = np.zeros(self.size, dtype=np.float32)

        return images

    def step(self, dt):

        self.graphicsEngine.renderFrame()

        # NOTE: we need to call frame rendering twice in onscreen mode because
        # of double-buffering
        if self.mode == 'onscreen':
            self.graphicsEngine.renderFrame()

    def getRenderInfo(self):
        sga = SceneGraphAnalyzer()
        sga.addNode(self.scene.scene.node())

        ls = LineStream()
        sga.write(ls)
        desc = []
        while ls.isTextAvailable():
            desc.append(ls.getLine())
        desc = '\n'.join(desc)
        return desc

    def _addDefaultLighting(self):
        alight = AmbientLight('alight')
        alight.setColor(VBase4(0.2, 0.2, 0.2, 1))
        alnp = self.scene.scene.attachNewNode(alight)
        self.scene.scene.setLight(alnp)

        for camera in self.cameras:

            # NOTE: Point light following the camera
            plight = PointLight('plight')
            plight.setColor(VBase4(1.0, 1.0, 1.0, 1))
            plnp = camera.attachNewNode(plight)
            self.scene.scene.setLight(plnp)

            if self.shadowing:
                # Use a 512x512 resolution shadow map
                plight.setShadowCaster(True, 512, 512)

                # Enable the shader generator for the receiving nodes
                self.scene.scene.setShaderAuto()
                self.scene.scene.setAntialias(AntialiasAttrib.MAuto)

        if self.modelLightsInfo is not None:

            # Add model-related lights (e.g. lamps)
            for model in self.scene.scene.findAllMatches('**/+ModelNode'):
                modelId = model.getNetTag('model-id')
                for lightNp in self.modelLightsInfo.getLightsForModel(modelId):

                    if self.shadowing:
                        # Use a 512x512 resolution shadow map
                        lightNp.node().setShadowCaster(True, 512, 512)

                    lightNp.reparentTo(model)

                    self.scene.scene.setLight(lightNp)


class Panda3dSemanticsRenderer(World):
    def __init__(self, scene, suncgDatasetRoot, size=(512, 512), mode='offscreen', zNear=0.1, zFar=1000.0, fov=40.0,
                 cameraTransform=None, segment_by_instance=False):

        super(Panda3dSemanticsRenderer, self).__init__()

        self.__dict__.update(scene=scene, suncgDatasetRoot=suncgDatasetRoot, size=size, mode=mode, zNear=zNear,
                             zFar=zFar, fov=fov,
                             cameraTransform=cameraTransform)

        self.categoryMapping = ModelCategoryMapping(
            os.path.join(
                self.suncgDatasetRoot,
                'metadata',
                'ModelCategoryMapping.csv')
        )

        self.cameraMask = BitMask32.bit(1)
        self.graphicsEngine = GraphicsEngine.getGlobalPtr()
        self.loader = Loader.getGlobalPtr()
        self.graphicsEngine.setDefaultLoader(self.loader)

        self.instance_color_mapping = {
            'ceiling': [153, 204, 204],
            'floor': [51, 51, 204],
            'wall': [102, 153, 255]
        }

        self.segment_by_instance = segment_by_instance

        self._initModels()

        selection = GraphicsPipeSelection.getGlobalPtr()
        self.pipe = selection.makeDefaultPipe()
        logger.debug('Using %s' % (self.pipe.getInterfaceName()))

        # Attach a camera to every agent in the scene
        self.cameras = []
        for agentNp in self.scene.scene.findAllMatches('**/agents/agent*'):
            camera = agentNp.attachNewNode(ModelNode('camera-semantics'))
            if self.cameraTransform is not None:
                camera.setTransform(cameraTransform)
            camera.node().setPreserveTransform(ModelNode.PTLocal)
            self.cameras.append(camera)

            # Reparent node below the existing physic node (if any)
            physicsNp = agentNp.find('**/physics')
            if not physicsNp.isEmpty():
                camera.reparentTo(physicsNp)

        self.rgbBuffers = dict()
        self.rgbTextures = dict()

        self._initRgbCapture()

        self.scene.worlds['render-semantics'] = self

    def _initCategoryColors(self):

        catNames = self.categoryMapping.getFineGrainedClassList()
        size = int(np.ceil(np.cbrt(len(catNames)) - 1e-6))

        # Uniform sampling of colors
        colors = np.zeros((size ** 3, 3))
        i = 0
        for r in np.linspace(0.0, 1.0, size):
            for g in np.linspace(0.0, 1.0, size):
                for b in np.linspace(0.0, 1.0, size):
                    colors[i] = [r, g, b]
                    i += 1

        # Shuffle
        indices = np.arange(len(colors))
        np.random.shuffle(indices)
        colors = colors[indices]
        self.catColors = dict()
        for i, name in enumerate(catNames):
            self.catColors[name] = colors[i]

            print('\'%s\': [%d, %d, %d],' % (name,
                                             int(colors[i][0] * 255),
                                             int(colors[i][1] * 255),
                                             int(colors[i][2] * 255)))

    def _initModels(self):

        models = []
        for model in self.scene.scene.findAllMatches('**/objects/**/+ModelNode'):
            models.append(model)
        for model in self.scene.scene.findAllMatches('**/layouts/**/+ModelNode'):
            models.append(model)

        for model in models:

            objectNp = model.getParent()
            rendererNp = objectNp.attachNewNode('render-semantics')
            model = model.copyTo(rendererNp)

            # Set the model to be visible only to this camera
            model.hide(BitMask32.allOn())
            model.show(self.cameraMask)

            # Get semantic-related color of model
            modelId = model.getNetTag('model-id')

            if self.segment_by_instance:
                instance_id = objectNp.getTag('instance-id')
                instance_color = None

            if 'fr_' in modelId:
                if modelId.endswith('c'):
                    catName = 'ceiling'
                elif modelId.endswith('f'):
                    catName = 'floor'
                elif modelId.endswith('w'):
                    catName = 'wall'

                if self.segment_by_instance:
                    instance_color = self.instance_color_mapping[catName]

            else:
                catName = self.categoryMapping.getFineGrainedCategoryForModelId(
                    modelId)

            color = MODEL_CATEGORY_COLOR_MAPPING[catName]

            if self.segment_by_instance:
                if not instance_color:
                    instance_color = list(np.random.choice(range(256), size=3))
                self.instance_color_mapping[instance_id] = instance_color
                color = instance_color

            # Clear all GeomNode render attributes and set a specified flat
            # color
            for nodePath in model.findAllMatches('**/+GeomNode'):
                geomNode = nodePath.node()
                for n in range(geomNode.getNumGeoms()):
                    geomNode.setGeomState(n, RenderState.make(
                        ColorAttrib.makeFlat(LColor(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0)), 1))

            self.color_instance_mapping = {
                tuple(v): k for k, v in six.iteritems(self.instance_color_mapping)}

            # Disable lights for this model
            model.setLightOff(1)

            # Enable antialiasing
            model.setAntialias(AntialiasAttrib.MAuto)

            # Reparent render node below the existing physic node (if any)
            physicsNp = objectNp.find('**/physics')
            if not physicsNp.isEmpty():
                rendererNp.reparentTo(physicsNp)

    def _initRgbCapture(self):

        for camera in self.cameras:

            camNode = Camera('Semantic camera')
            camNode.setCameraMask(self.cameraMask)
            lens = PerspectiveLens()
            lens.setFov(self.fov)
            lens.setAspectRatio(float(self.size[0]) / float(self.size[1]))
            lens.setNear(self.zNear)
            lens.setFar(self.zFar)
            camNode.setLens(lens)
            camNode.setScene(self.scene.scene)
            cam = camera.attachNewNode(camNode)

            winprops = WindowProperties.size(self.size[0], self.size[1])
            fbprops = FrameBufferProperties.getDefault()
            fbprops = FrameBufferProperties(fbprops)
            fbprops.setRgbaBits(8, 8, 8, 8)

            flags = GraphicsPipe.BFFbPropsOptional
            if self.mode == 'onscreen':
                flags = flags | GraphicsPipe.BFRequireWindow
            elif self.mode == 'offscreen':
                flags = flags | GraphicsPipe.BFRefuseWindow
            else:
                raise Exception('Unsupported rendering mode: %s' % (self.mode))

            buf = self.graphicsEngine.makeOutput(self.pipe, 'RGB-buffer-Semantics', 0, fbprops,
                                                 winprops, flags)
            if buf is None:
                raise Exception('Unable to create RGB buffer')

            # Set to render at the end
            buf.setSort(10000)

            dr = buf.makeDisplayRegion()
            dr.setSort(0)
            dr.setCamera(cam)
            dr = camNode.getDisplayRegion(0)

            tex = Texture()
            tex.setFormat(Texture.FRgba8)
            tex.setComponentType(Texture.TUnsignedByte)
            buf.addRenderTexture(
                tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)
            # XXX: should use tex.setMatchFramebufferFormat(True)?

            self.rgbBuffers[camera.getNetTag('agent-id')] = buf
            self.rgbTextures[camera.getNetTag('agent-id')] = tex

    def showRoomLayout(self, showCeilings=True, showWalls=True, showFloors=True):

        for np in self.scene.scene.findAllMatches('**/layouts/**/render-semantics/*c'):
            if showCeilings:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

        for np in self.scene.scene.findAllMatches('**/layouts/**/render-semantics/*w'):
            if showWalls:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

        for np in self.scene.scene.findAllMatches('**/layouts/**/render-semantics/*f'):
            if showFloors:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

    def destroy(self):
        self.graphicsEngine.removeAllWindows()
        del self.pipe

    def getRgbaImages(self, channelOrder="RGBA"):
        images = dict()
        for name, tex in six.iteritems(self.rgbTextures):
            # XXX: not sure about calling makeRamImage() before getting the image data, since it returns an empty image
            # and overwrite any previously rendered image. We may just call it
            # once when we create the texture.
            data = tex.getRamImageAs(channelOrder)
            if not tex.mightHaveRamImage():
                tex.makeRamImage()
                data = tex.getRamImageAs(channelOrder)
            try:
                data_img = data.get_data()
            except UnicodeDecodeError:
                tex.makeRamImage()
                data = tex.getRamImageAs(channelOrder)
                data_img = data.get_data()
            if (sys.version_info > (3, 0)):
                # Must match Texture.TUnsignedByte
                image = np.fromstring(data_img, dtype=np.uint8)
            else:
                # Must match Texture.TUnsignedByte
                image = np.frombuffer(data_img, dtype=np.uint8)
            image.shape = (tex.getYSize(), tex.getXSize(), 4)
            image = np.flipud(image)
            images[name] = image

        return images

    def step(self, dt):

        self.graphicsEngine.renderFrame()

        # NOTE: we need to call frame rendering twice in onscreen mode because
        # of double-buffering
        if self.mode == 'onscreen':
            self.graphicsEngine.renderFrame()


class SemanticRenderer(object):

    def __init__(self, scene, size=(512, 512), mode='offscreen',
                 zNear=0.1, zFar=1000.0, fov=40.0,
                 cameraTransform=None):

        # Off-screen buffers are not supported in OSX
        if sys.platform == 'darwin':
            mode = 'onscreen'

        super(SemanticRenderer, self).__init__()

        self.__dict__.update(scene=scene, size=size, mode=mode, zNear=zNear,
                             zFar=zFar, fov=fov,
                             cameraTransform=cameraTransform)
        self.categoryColors = dict()
        self.instanceColors = dict()

        self.cameraMask = BitMask32.bit(1)
        self.graphicsEngine = GraphicsEngine.getGlobalPtr()
        self.loader = Loader.getGlobalPtr()
        self.graphicsEngine.setDefaultLoader(self.loader)

        self._initModels()

        selection = GraphicsPipeSelection.getGlobalPtr()
        self.pipe = selection.makeDefaultPipe()
        logger.debug('Using %s' % (self.pipe.getInterfaceName()))

        # Attach a camera to every agent in the scene
        self.cameras = []
        for agentNp in self.scene.scene.findAllMatches('**/agents/agent*'):
            camera = agentNp.attachNewNode(ModelNode('render-semantic'))
            if self.cameraTransform is not None:
                camera.setTransform(cameraTransform)
            camera.node().setPreserveTransform(ModelNode.PTLocal)
            self.cameras.append(camera)

            # Reparent node below the existing physic node (if any)
            physicsNp = agentNp.find('**/physics')
            if not physicsNp.isEmpty():
                camera.reparentTo(physicsNp)

        self.rgbBuffers = dict()
        self.rgbTextures = dict()

        self._initRgbCapture()

    def _initInstanceColors(self, models, shuffle=False):
        try:
            SUNCG_DATA_DIR = os.environ["SUNCG_DATA_DIR"]
        except KeyError:
            raise Exception("Please set the environment variable SUNCG_DATA_DIR")

        categoryMapping = ModelCategoryMapping(
            os.path.join(
                SUNCG_DATA_DIR,
                'metadata',
                'ModelCategoryMapping.csv'))

        catNames = categoryMapping.getFineGrainedClassList()
        size = int(np.ceil(np.cbrt(len(catNames)) - 1e-6))

        # Uniform sampling of colors
        colors = np.zeros((size ** 3, 3))
        i = 0
        for r in np.linspace(0.0, 1.0, size):
            for g in np.linspace(0.0, 1.0, size):
                for b in np.linspace(0.0, 1.0, size):
                    colors[i] = [r, g, b]
                    i += 1

        if shuffle:
            # Shuffle
            indices = np.arange(len(colors))
            np.random.shuffle(indices)
            colors = colors[indices]

        categoryColors = dict()
        for catName, color in zip(catNames, colors):
            categoryColors[catName] = color

        instanceColors = dict()
        for model in models:
            modelId = model.getNetTag('model-id')
            if 'fr_' in modelId:
                if modelId.endswith('c'):
                    catName = 'ceiling'
                elif modelId.endswith('f'):
                    catName = 'floor'
                elif modelId.endswith('w'):
                    catName = 'wall'
            else:
                catName = categoryMapping.getFineGrainedCategoryForModelId(modelId)

            instanceId = model.getNetTag('instance-id')
            instanceColors[instanceId] = colors[catNames.index(catName)]

        return instanceColors, categoryColors

    def _initModels(self):

        # Get the list of all models in the scene
        models = []
        for model in self.scene.scene.findAllMatches('**/objects/*/+ModelNode'):
            models.append(model)
        for model in self.scene.scene.findAllMatches('**/layouts/*/+ModelNode'):
            models.append(model)

        # Associate a color with each model
        self.instanceColors, self.categoryColors = self._initInstanceColors(models)

        for model in models:

            objectNp = model.getParent()
            rendererNp = objectNp.attachNewNode('render-semantic')
            model = model.copyTo(rendererNp)

            # Set the model to be visible only to this camera
            model.hide(BitMask32.allOn())
            model.show(self.cameraMask)

            # Get semantic-related color of model
            instanceId = model.getNetTag('instance-id')
            color = self.instanceColors[instanceId]

            # Clear all GeomNode render attributes and set a specified flat
            # color
            for nodePath in model.findAllMatches('**/+GeomNode'):
                geomNode = nodePath.node()
                for n in range(geomNode.getNumGeoms()):
                    geomNode.setGeomState(n, RenderState.make(
                        ColorAttrib.makeFlat(LColor(color[0], color[1], color[2], 1.0)), 1))

            # Disable lights for this model
            model.setLightOff(1)

            # Enable antialiasing
            model.setAntialias(AntialiasAttrib.MAuto)

            # Reparent render node below the existing physic node (if any)
            physicsNp = objectNp.find('**/physics')
            if not physicsNp.isEmpty():
                rendererNp.reparentTo(physicsNp)

    def _initRgbCapture(self):

        for camera in self.cameras:

            camNode = Camera('Semantic rendering camera')
            camNode.setCameraMask(self.cameraMask)
            lens = PerspectiveLens()
            lens.setFov(self.fov)
            lens.setAspectRatio(float(self.size[0]) / float(self.size[1]))
            lens.setNear(self.zNear)
            lens.setFar(self.zFar)
            camNode.setLens(lens)
            camNode.setScene(self.scene.scene)
            cam = camera.attachNewNode(camNode)

            winprops = WindowProperties.size(self.size[0], self.size[1])
            fbprops = FrameBufferProperties.getDefault()
            fbprops = FrameBufferProperties(fbprops)
            fbprops.setRgbaBits(8, 8, 8, 8)

            flags = GraphicsPipe.BFFbPropsOptional
            if self.mode == 'onscreen':
                flags = flags | GraphicsPipe.BFRequireWindow
            elif self.mode == 'offscreen':
                flags = flags | GraphicsPipe.BFRefuseWindow
            else:
                raise Exception('Unsupported rendering mode: %s' % (self.mode))

            buf = self.graphicsEngine.makeOutput(self.pipe, 'RGB buffer - semantic rendering', 0, fbprops,
                                                 winprops, flags)
            if buf is None:
                raise Exception('Unable to create RGB buffer')

            # Set to render at the end
            buf.setSort(10000)

            dr = buf.makeDisplayRegion()
            dr.setSort(0)
            dr.setCamera(cam)
            dr = camNode.getDisplayRegion(0)

            tex = Texture()
            tex.setFormat(Texture.FRgba8)
            tex.setComponentType(Texture.TUnsignedByte)
            buf.addRenderTexture(
                tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)
            tex.makeRamImage()
            # XXX: should use tex.setMatchFramebufferFormat(True)?

            self.rgbBuffers[camera.getNetTag('agent-id')] = buf
            self.rgbTextures[camera.getNetTag('agent-id')] = tex

    def destroy(self):
        self.graphicsEngine.removeAllWindows()
        del self.pipe

    def getSemanticImage(self, agentId, channelOrder="RGBA"):

        self.graphicsEngine.renderFrame()

        # NOTE: we need to call frame rendering twice in onscreen mode because
        # of double-buffering
        if self.mode == 'onscreen':
            self.graphicsEngine.renderFrame()

        tex = self.rgbTextures[agentId]
        data = tex.getRamImageAs(channelOrder)
        if (sys.version_info > (3, 0)):
            # Must match Texture.TUnsignedByte
            # NOTE: see https://github.com/panda3d/panda3d/issues/173
            data_img = bytes(memoryview(data))
            image = np.frombuffer(data_img, dtype=np.uint8)
        else:
            # Must match Texture.TUnsignedByte
            data_img = data.get_data()
            image = np.frombuffer(data_img, dtype=np.uint8)
        image.shape = (tex.getYSize(), tex.getXSize(), 4)
        image = np.flipud(image)

        return image


class InstancesRenderer(object):

    def __init__(self, scene, size=(512, 512), mode='offscreen',
                 zNear=0.1, zFar=1000.0, fov=40.0,
                 cameraTransform=None):

        # Off-screen buffers are not supported in OSX
        if sys.platform == 'darwin':
            mode = 'onscreen'

        super(InstancesRenderer, self).__init__()

        instanceColors = dict()
        self.__dict__.update(scene=scene, size=size, mode=mode, zNear=zNear,
                             zFar=zFar, fov=fov,
                             cameraTransform=cameraTransform, instanceColors=instanceColors)

        self.cameraMask = BitMask32.bit(7)
        self.graphicsEngine = GraphicsEngine.getGlobalPtr()
        self.loader = Loader.getGlobalPtr()
        self.graphicsEngine.setDefaultLoader(self.loader)

        self._initModels()

        selection = GraphicsPipeSelection.getGlobalPtr()
        self.pipe = selection.makeDefaultPipe()
        logger.debug('Using %s' % (self.pipe.getInterfaceName()))

        # Attach a camera to every agent in the scene
        self.cameras = []
        for agentNp in self.scene.scene.findAllMatches('**/agents/agent*'):
            camera = agentNp.attachNewNode(ModelNode('render-instances'))
            if self.cameraTransform is not None:
                camera.setTransform(cameraTransform)
            camera.node().setPreserveTransform(ModelNode.PTLocal)
            self.cameras.append(camera)

            # Reparent node below the existing physic node (if any)
            physicsNp = agentNp.find('**/physics')
            if not physicsNp.isEmpty():
                camera.reparentTo(physicsNp)

        self.rgbBuffers = dict()
        self.rgbTextures = dict()

        self._initRgbCapture()

    def _initInstanceColors(self, models):

        size = int(np.ceil(np.cbrt(len(models)) - 1e-6))

        # Uniform sampling of colors
        colors = np.zeros((size ** 3, 4))
        i = 0
        for r in np.linspace(0.0, 1.0, size):
            for g in np.linspace(0.0, 1.0, size):
                for b in np.linspace(0.0, 1.0, size):
                    colors[i] = [r, g, b, 1.0]
                    i += 1

        # Shuffle
        indices = np.arange(len(colors))
        np.random.shuffle(indices)
        colors = colors[indices]

        instanceColors = dict()
        for i, model in enumerate(models):
            instanceId = model.getNetTag('instance-id')
            instanceColors[instanceId] = colors[i]

        return instanceColors

    def _initModels(self):

        # Get the list of all models in the scene
        models = []
        for model in self.scene.scene.findAllMatches('**/objects/*/+ModelNode'):
            models.append(model)
        for model in self.scene.scene.findAllMatches('**/layouts/*/+ModelNode'):
            models.append(model)

        # Associate a color with each model
        self.instanceColors = self._initInstanceColors(models)

        for model in models:

            objectNp = model.getParent()
            rendererNp = objectNp.attachNewNode('render-instances')
            model = model.copyTo(rendererNp)

            # Set the model to be visible only to this camera
            model.hide(BitMask32.allOn())
            model.show(self.cameraMask)

            # Get semantic-related color of model
            instanceId = model.getNetTag('instance-id')
            color = self.instanceColors[instanceId]

            # Clear all GeomNode render attributes and set a specified flat
            # color
            for nodePath in model.findAllMatches('**/+GeomNode'):
                geomNode = nodePath.node()
                for n in range(geomNode.getNumGeoms()):
                    geomNode.setGeomState(n, RenderState.make(
                        ColorAttrib.makeFlat(LColor(color[0], color[1], color[2], 1.0)), 1))

            # Disable lights for this model
            model.setLightOff(1)

            # Enable antialiasing
            model.setAntialias(AntialiasAttrib.MAuto)

            # Reparent render node below the existing physic node (if any)
            physicsNp = objectNp.find('**/physics')
            if not physicsNp.isEmpty():
                rendererNp.reparentTo(physicsNp)

    def _initRgbCapture(self):

        for camera in self.cameras:

            camNode = Camera('Instances rendering camera')
            camNode.setCameraMask(self.cameraMask)
            lens = PerspectiveLens()
            lens.setFov(self.fov)
            lens.setAspectRatio(float(self.size[0]) / float(self.size[1]))
            lens.setNear(self.zNear)
            lens.setFar(self.zFar)
            camNode.setLens(lens)
            camNode.setScene(self.scene.scene)
            cam = camera.attachNewNode(camNode)

            winprops = WindowProperties.size(self.size[0], self.size[1])
            fbprops = FrameBufferProperties.getDefault()
            fbprops = FrameBufferProperties(fbprops)
            fbprops.setRgbaBits(8, 8, 8, 8)

            flags = GraphicsPipe.BFFbPropsOptional
            if self.mode == 'onscreen':
                flags = flags | GraphicsPipe.BFRequireWindow
            elif self.mode == 'offscreen':
                flags = flags | GraphicsPipe.BFRefuseWindow
            else:
                raise Exception('Unsupported rendering mode: %s' % (self.mode))

            buf = self.graphicsEngine.makeOutput(self.pipe, 'RGB buffer - instances rendering', 0, fbprops,
                                                 winprops, flags)
            if buf is None:
                raise Exception('Unable to create RGB buffer')

            # Set to render at the end
            buf.setSort(10000)

            dr = buf.makeDisplayRegion()
            dr.setSort(0)
            dr.setCamera(cam)
            dr = camNode.getDisplayRegion(0)

            tex = Texture()
            tex.setFormat(Texture.FRgba8)
            tex.setComponentType(Texture.TUnsignedByte)
            buf.addRenderTexture(
                tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)
            tex.makeRamImage()
            # XXX: should use tex.setMatchFramebufferFormat(True)?

            self.rgbBuffers[camera.getNetTag('agent-id')] = buf
            self.rgbTextures[camera.getNetTag('agent-id')] = tex

    def destroy(self):
        self.graphicsEngine.removeAllWindows()
        del self.pipe

    def _getVisibleSurfaceForObjectId(self, image, instanceId):
        uniqueColors, counts = np.unique(
            image.reshape(-1, image.shape[2]), axis=0, return_counts=True)
        uniqueColors = uniqueColors / 255.0

        surface = 0.0
        refColor = self.instanceColors[instanceId]
        for color, count in zip(uniqueColors, counts):
            if np.allclose(color, refColor, atol=1 / 255.0):
                surface = float(count)

        return surface

    def getOccludedObjectIds(self, agentId):

        occlusionRatios = dict()
        visibleInstanceIds = self.getVisibleObjectIds(agentId)
        for refInstanceId in visibleInstanceIds:

            # Get the visible surface with occlusion
            image = self.getInstancesImage(agentId)
            surfaceOccluded = self._getVisibleSurfaceForObjectId(
                image, refInstanceId)

            # Hide all other models in the scene but his one
            for model in self.scene.scene.findAllMatches('**/objects/*/render-instances/+ModelNode'):
                instanceId = model.getNetTag('instance-id')
                if instanceId != refInstanceId and instanceId in visibleInstanceIds:
                    model.hide(BitMask32(self.cameraMask))
            for model in self.scene.scene.findAllMatches('**/layouts/*/render-instances/+ModelNode'):
                instanceId = model.getNetTag('instance-id')
                if instanceId != refInstanceId and instanceId in visibleInstanceIds:
                    model.hide(BitMask32(self.cameraMask))

            # Get the visible surface without occlusion
            image = self.getInstancesImage(agentId)
            surfaceRef = self._getVisibleSurfaceForObjectId(
                image, refInstanceId)

            # Revert the scene to original
            for model in self.scene.scene.findAllMatches('**/objects/*/render-instances/+ModelNode'):
                instanceId = model.getNetTag('instance-id')
                if model.getNetTag('instance-id') != refInstanceId:
                    model.show(BitMask32(self.cameraMask))
            for model in self.scene.scene.findAllMatches('**/layouts/*/render-instances/+ModelNode'):
                instanceId = model.getNetTag('instance-id')
                if model.getNetTag('instance-id') != refInstanceId:
                    model.show(BitMask32(self.cameraMask))

            # Calculate the ratio of occlusion
            occlusionRatio = 1.0 - surfaceOccluded / surfaceRef
            assert occlusionRatio >= 0.0 and occlusionRatio <= 1.0

            occlusionRatios[refInstanceId] = occlusionRatio

        return occlusionRatios

    def getVisibleObjectIds(self, agentId):

        # Get the list of visible objects
        image = self.getInstancesImage(agentId)

        uniqueColors = np.unique(
            image.reshape(-1, image.shape[2]), axis=0) / 255.0
        visibleModelIds = []
        for color in uniqueColors:
            for instanceId, refColor in six.iteritems(self.instanceColors):
                if np.allclose(color, refColor, atol=1 / 255.0):
                    visibleModelIds.append(instanceId)

        return visibleModelIds

    def getInstancesImage(self, agentId, channelOrder="RGBA"):

        self.graphicsEngine.renderFrame()

        # NOTE: we need to call frame rendering twice in onscreen mode because
        # of double-buffering
        if self.mode == 'onscreen':
            self.graphicsEngine.renderFrame()

        tex = self.rgbTextures[agentId]
        data = tex.getRamImageAs(channelOrder)
        if (sys.version_info > (3, 0)):
            # Must match Texture.TUnsignedByte
            # NOTE: see https://github.com/panda3d/panda3d/issues/173
            data_img = bytes(memoryview(data))
            image = np.frombuffer(data_img, dtype=np.uint8)
        else:
            # Must match Texture.TUnsignedByte
            data_img = data.get_data()
            image = np.frombuffer(data_img, dtype=np.uint8)
        image.shape = (tex.getYSize(), tex.getXSize(), 4)
        image = np.flipud(image)

        return image


class RgbRenderer(object):
    def __init__(self, scene, size=(512, 512), mode='offscreen', zNear=0.1, zFar=1000.0, fov=40.0, cameraTransform=None):

        # Off-screen buffers are not supported in OSX
        if sys.platform == 'darwin':
            mode = 'onscreen'

        super(RgbRenderer, self).__init__()

        self.__dict__.update(scene=scene, size=size, mode=mode, zNear=zNear, zFar=zFar, fov=fov,
                             cameraTransform=cameraTransform)

        self.cameraMask = BitMask32.bit(0)
        self.graphicsEngine = GraphicsEngine.getGlobalPtr()
        self.loader = Loader.getGlobalPtr()
        self.graphicsEngine.setDefaultLoader(self.loader)

        # Change some scene attributes for rendering
        self.scene.scene.setAttrib(RescaleNormalAttrib.makeDefault())
        self.scene.scene.setTwoSided(True)

        selection = GraphicsPipeSelection.getGlobalPtr()
        self.pipe = selection.makeDefaultPipe()
        logger.debug('Using %s' % (self.pipe.getInterfaceName()))

        # Attach a camera to every agent in the scene
        self.cameras = []
        for agentNp in self.scene.scene.findAllMatches('**/agents/agent*'):
#            a=agentNp.getChild(0).getChild(1) ## point it to the Neck
            a=agentNp
            camera = a.attachNewNode(ModelNode('camera-rgb'))
            if self.cameraTransform is not None:
                camera.setTransform(cameraTransform)
#            camera.setPos(0,-0.5,2)
#            camera.setHpr(180,0,0)
            camera.reparentTo(a)
            camera.node().setPreserveTransform(ModelNode.PTLocal)
            self.cameras.append(camera)

        self.rgbBuffers = dict()
        self.rgbTextures = dict()

        self._initRgbCapture()
        self._addDefaultLighting()

        self.notifySceneChanged()

    def notifySceneChanged(self):

        for modelNp in self.scene.scene.findAllMatches('**/model-*'):

            isInitialized = False
            objectNp = modelNp.getParent()
            for childNp in modelNp.getChildren():
                if childNp.getName() == 'render-rgb':
                    isInitialized = True
                    break

            if not isInitialized:
                rendererNp = objectNp.attachNewNode('render-rgb')
                model = modelNp.copyTo(rendererNp)

                # Set the model to be visible only to this camera
                model.hide(BitMask32.allOn())
                model.show(self.cameraMask)

                # Reparent render node below the existing physic node (if any)
                physicsNp = objectNp.find('**/physics')
                if not physicsNp.isEmpty():
                    rendererNp.reparentTo(physicsNp)

    def setBackgroundColor(self, rgba):
        for buf in six.itervalues(self.rgbBuffers):
            buf.setClearColor(LVector4f(*rgba))

    def _initRgbCapture(self):

        for camera in self.cameras:

            camNode = Camera('RGB camera')
            camNode.setCameraMask(self.cameraMask)
            lens = PerspectiveLens()
            lens.setFov(self.fov)
            lens.setAspectRatio(float(self.size[0]) / float(self.size[1]))
            lens.setNear(self.zNear)
            lens.setFar(self.zFar)
            camNode.setLens(lens)
            camNode.setScene(self.scene.scene)
            cam = camera.attachNewNode(camNode)

            winprops = WindowProperties.size(self.size[0], self.size[1])
            fbprops = FrameBufferProperties.getDefault()
            fbprops = FrameBufferProperties(fbprops)
            fbprops.setRgbaBits(8, 8, 8, 0)

            flags = GraphicsPipe.BFFbPropsOptional
            if self.mode == 'onscreen':
                flags = flags | GraphicsPipe.BFRequireWindow
            elif self.mode == 'offscreen':
                flags = flags | GraphicsPipe.BFRefuseWindow
            else:
                raise Exception('Unsupported rendering mode: %s' % (self.mode))

            buf = self.graphicsEngine.makeOutput(self.pipe, 'RGB buffer Rendering', 0, fbprops,
                                                 winprops, flags)
            if buf is None:
                raise Exception('Unable to create RGB buffer')

            # Set to render at the end
            buf.setSort(10000)

            dr = buf.makeDisplayRegion()
            dr.setSort(0)
            dr.setCamera(cam)
            dr = camNode.getDisplayRegion(0)

            tex = Texture()
            tex.setFormat(Texture.FRgb8)
            tex.setComponentType(Texture.TUnsignedByte)
            buf.addRenderTexture(
                tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)
            tex.makeRamImage()
            # XXX: should use tex.setMatchFramebufferFormat(True)?
            # agent=camera.getParent() # This is when camer is mounted on agent
            agent = camera.getParent()
            self.rgbBuffers[agent.getName()] = buf
            self.rgbTextures[agent.getName()] = tex

    def showRoomLayout(self, showCeilings=True, showWalls=True, showFloors=True):

        for np in self.scene.scene.findAllMatches('**/layouts/**/render-rgb/*c'):	
            if showCeilings:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

        for np in self.scene.scene.findAllMatches('**/layouts/**/render-rgb/*w'):
            if showWalls:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

        for np in self.scene.scene.findAllMatches('**/layouts/**/render-rgb/*f'):
            if showFloors:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

    def destroy(self):
        self.graphicsEngine.removeAllWindows()
        del self.pipe

    def getRgbImage(self, agentId, channelOrder="RGB"):

        self.graphicsEngine.renderFrame()

        # NOTE: we need to call frame rendering twice in onscreen mode because
        # of double-buffering
        if self.mode == 'onscreen':
            self.graphicsEngine.renderFrame()
        tex = self.rgbTextures[agentId]

        # XXX: not sure about calling makeRamImage() before getting the image data, since it returns an empty image
        # and overwrite any previously rendered image. We may just call it
        # once when we create the texture.
        if not tex.mightHaveRamImage():
            tex.makeRamImage()

        if sys.version_info[0] < 3:
            data = tex.getRamImageAs(channelOrder).getData()   # Python 2
        else:
            # NOTE: see https://github.com/panda3d/panda3d/issues/173
            data = bytes(memoryview(
                tex.getRamImageAs(channelOrder)))  # Python 3

        # Must match Texture.TUnsignedByte
        image = np.frombuffer(data, dtype=np.uint8)

        image.shape = (tex.getYSize(), tex.getXSize(), len(channelOrder))
        image = np.flipud(image)

        return image

    def _addDefaultLighting(self):
        alight = AmbientLight('alight')
        alight.setColor(LVector4f(0.2, 0.2, 0.2, 1))
        alnp = self.scene.scene.attachNewNode(alight)
        self.scene.scene.setLight(alnp)

        for camera in self.cameras:

            # NOTE: Point light following the camera
            plight = PointLight('plight')
            plight.setColor(LVector4f(1.0, 1.0, 1.0, 1))
            plnp = camera.attachNewNode(plight)
            self.scene.scene.setLight(plnp)


class DepthRenderer(object):
    def __init__(self, scene, size=(512, 512), mode='offscreen', zNear=0.1, zFar=1000.0, fov=40.0, cameraTransform=None):

        # Off-screen buffers are not supported in OSX
        if sys.platform == 'darwin':
            mode = 'onscreen'

        self.__dict__.update(scene=scene, size=size, mode=mode, zNear=zNear, zFar=zFar, fov=fov,
                             cameraTransform=cameraTransform)

        self.cameraMask = BitMask32.bit(9)
        self.graphicsEngine = GraphicsEngine.getGlobalPtr()
        self.loader = Loader.getGlobalPtr()
        self.graphicsEngine.setDefaultLoader(self.loader)

        self._initModels()

        selection = GraphicsPipeSelection.getGlobalPtr()
        self.pipe = selection.makeDefaultPipe()
        logger.debug('Using %s' % (self.pipe.getInterfaceName()))

        # Attach a camera to every agent in the scene
        self.cameras = []
        for agentNp in self.scene.scene.findAllMatches('**/agents/agent*'):
            camera = agentNp.attachNewNode(ModelNode('camera-depth'))
            if self.cameraTransform is not None:
                camera.setTransform(cameraTransform)
            camera.node().setPreserveTransform(ModelNode.PTLocal)
            self.cameras.append(camera)

        self.depthBuffers = dict()
        self.depthTextures = dict()

        self._initDepthCapture()

    def _initModels(self):

        for model in self.scene.scene.findAllMatches('**/+ModelNode'):

            objectNp = model.getParent()
            rendererNp = objectNp.attachNewNode('render-depth')
            model = model.copyTo(rendererNp)

            # Set the model to be visible only to this camera
            model.hide(BitMask32.allOn())
            model.show(self.cameraMask)

            # Reparent render node below the existing physic node (if any)
            physicsNp = objectNp.find('**/physics')
            if not physicsNp.isEmpty():
                rendererNp.reparentTo(physicsNp)

    def _initDepthCapture(self):

        for camera in self.cameras:

            camNode = Camera('Depth camera')
            camNode.setCameraMask(self.cameraMask)
            lens = PerspectiveLens()
            lens.setFov(self.fov)
            lens.setAspectRatio(float(self.size[0]) / float(self.size[1]))
            lens.setNear(self.zNear)
            lens.setFar(self.zFar)
            camNode.setLens(lens)
            camNode.setScene(self.scene.scene)
            cam = camera.attachNewNode(camNode)

            winprops = WindowProperties.size(self.size[0], self.size[1])
            fbprops = FrameBufferProperties.getDefault()
            fbprops = FrameBufferProperties(fbprops)
            fbprops.setRgbColor(False)
            fbprops.setRgbaBits(0, 0, 0, 0)
            fbprops.setStencilBits(0)
            fbprops.setMultisamples(0)
            fbprops.setBackBuffers(0)
            fbprops.setDepthBits(16)

            flags = GraphicsPipe.BFFbPropsOptional
            if self.mode == 'onscreen':
                flags = flags | GraphicsPipe.BFRequireWindow
            elif self.mode == 'offscreen':
                flags = flags | GraphicsPipe.BFRefuseWindow
            else:
                raise Exception('Unsupported rendering mode: %s' % (self.mode))

            buf = self.graphicsEngine.makeOutput(self.pipe, 'Depth buffer', 0, fbprops,
                                                 winprops, flags)
            if buf is None:
                raise Exception('Unable to create depth buffer')

            # Set to render at the end
            buf.setSort(10000)

            dr = buf.makeDisplayRegion()
            dr.setSort(0)
            dr.setCamera(cam)
            dr = camNode.getDisplayRegion(0)

            tex = Texture()
            tex.setFormat(Texture.FDepthComponent)
            tex.setComponentType(Texture.TFloat)
            buf.addRenderTexture(
                tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)
            tex.makeRamImage()
            # XXX: should use tex.setMatchFramebufferFormat(True)?

            agentId = camera.getNetTag('agent-id')
            self.depthBuffers[agentId] = buf
            self.depthTextures[agentId] = tex

    def showRoomLayout(self, showCeilings=True, showWalls=True, showFloors=True):

        for np in self.scene.scene.findAllMatches('**/layouts/**/render-depth/*c'):
            if showCeilings:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

        for np in self.scene.scene.findAllMatches('**/layouts/**/render-depth/*w'):
            if showWalls:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

        for np in self.scene.scene.findAllMatches('**/layouts/**/render-depth/*f'):
            if showFloors:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

    def destroy(self):
        self.graphicsEngine.removeAllWindows()
        del self.pipe

    def getDepthImage(self, agentId, mode='normalized'):

        self.graphicsEngine.renderFrame()

        # NOTE: we need to call frame rendering twice in onscreen mode because
        # of double-buffering
        if self.mode == 'onscreen':
            self.graphicsEngine.renderFrame()

        tex = self.depthTextures[agentId]

        # XXX: not sure about calling makeRamImage() before getting the image data, since it returns an empty image
        # and overwrite any previously rendered image. We may just call it
        # once when we create the texture.
        if not tex.mightHaveRamImage():
            tex.makeRamImage()

        if sys.version_info[0] < 3:
            data = tex.getRamImage().getData()   # Python 2
        else:
            # NOTE: see https://github.com/panda3d/panda3d/issues/173
            data = bytes(memoryview(tex.getRamImage()))  # Python 3

        nbBytesComponentFromData = len(
            data) / (tex.getYSize() * tex.getXSize())
        if nbBytesComponentFromData == 4:
            # Must match Texture.TFloat
            depthImage = np.frombuffer(data, dtype=np.float32)

        elif nbBytesComponentFromData == 2:
            # NOTE: This can happen on some graphic hardware, where unsigned 16-bit data is stored
            # despite setting the texture component type to 32-bit floating
            # point.
            # Must match Texture.TFloat
            depthImage = np.frombuffer(data, dtype=np.uint16)
            depthImage = depthImage.astype(np.float32) / 65535

        depthImage.shape = (tex.getYSize(), tex.getXSize())
        depthImage = np.flipud(depthImage)

        if mode == 'distance':
            # NOTE: in Panda3d, the returned depth image seems to be
            # already linearized
            depthImage = self.zNear + depthImage / (self.zFar - self.zNear)

            # Adapted from: https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
            # depthImage = 2.0 * depthImage - 1.0
            # depthImage = 2.0 * self.zNear * self.zFar / (self.zFar + self.zNear - depthImage * (self.zFar - self.zNear))

        elif mode == 'normalized':
            # Nothing to do
            pass
        else:
            raise Exception(
                'Unsupported output depth image mode: %s' % (mode))

        return depthImage


def get3DPointsFromModel(model):
    geomNodes = model.findAllMatches('**/+GeomNode')

    pts = []
    for nodePath in geomNodes:
        nodePts = []
        geomNode = nodePath.node()
        for i in range(geomNode.getNumGeoms()):
            geom = geomNode.getGeom(i)
            vdata = geom.getVertexData()
            vertex = GeomVertexReader(vdata, 'vertex')
            while not vertex.isAtEnd():
                v = vertex.getData3f()
                nodePts.append([v.x, v.y, v.z])
        pts.append(nodePts)
    return np.array(pts)


def get3DTrianglesFromModel(model):
    # Calculate the net transformation
    transform = model.getNetTransform()
    transformMat = transform.getMat()

    # Get geometry data from GeomNode instances inside the model
    geomNodes = model.findAllMatches('**/+GeomNode')

    triangles = []
    for nodePath in geomNodes:
        geomNode = nodePath.node()

        for n in range(geomNode.getNumGeoms()):
            geom = geomNode.getGeom(n)
            vdata = geom.getVertexData()

            for k in range(geom.getNumPrimitives()):
                prim = geom.getPrimitive(k)
                vertex = GeomVertexReader(vdata, 'vertex')
                assert isinstance(prim, (GeomTristrips, GeomTriangles))

                # Decompose into triangles
                prim = prim.decompose()
                for p in range(prim.getNumPrimitives()):
                    s = prim.getPrimitiveStart(p)
                    e = prim.getPrimitiveEnd(p)

                    triPts = []
                    for i in range(s, e):
                        vi = prim.getVertex(i)
                        vertex.setRow(vi)
                        v = vertex.getData3f()

                        # Apply transformation
                        v = transformMat.xformPoint(v)

                        triPts.append([v.x, v.y, v.z])

                    triangles.append(triPts)

    triangles = np.array(triangles)

    return triangles


def getSurfaceAreaFromGeom(geom, transform=None):
    totalArea = 0.0
    for k in range(geom.getNumPrimitives()):
        prim = geom.getPrimitive(k)
        vdata = geom.getVertexData()
        vertex = GeomVertexReader(vdata, 'vertex')
        assert isinstance(prim, (GeomTristrips, GeomTriangles))

        # Decompose into triangles
        prim = prim.decompose()
        for p in range(prim.getNumPrimitives()):
            s = prim.getPrimitiveStart(p)
            e = prim.getPrimitiveEnd(p)

            triPts = []
            for i in range(s, e):
                vi = prim.getVertex(i)
                vertex.setRow(vi)
                v = vertex.getData3f()

                # Apply transformation
                if transform is not None:
                    v = transform.xformPoint(v)

                triPts.append([v.x, v.y, v.z])
            triPts = np.array(triPts)

            # calculate the semi-perimeter and area
            a = np.linalg.norm(triPts[0] - triPts[1], 2)
            b = np.linalg.norm(triPts[1] - triPts[2], 2)
            c = np.linalg.norm(triPts[2] - triPts[0], 2)
            s = (a + b + c) / 2
            area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
            totalArea += area

    return totalArea


def getColorAttributesFromVertexData(geom, transform=None):
    colorsTotalAreas = dict()
    for k in range(geom.getNumPrimitives()):
        prim = geom.getPrimitive(k)
        vdata = geom.getVertexData()
        assert isinstance(prim, (GeomTristrips, GeomTriangles))

        # Check if color is defined for vertex
        isColorDefined = False
        for i, geomVertexCol in enumerate(vdata.getFormat().getColumns()):
            if geomVertexCol.getContents() == GeomEnums.CColor:
                isColorDefined = True
                break
        assert isColorDefined

        vertex = GeomVertexReader(vdata, 'vertex')
        vertexColor = GeomVertexReader(vdata, 'color')

        # Decompose into triangles
        prim = prim.decompose()
        for p in range(prim.getNumPrimitives()):
            s = prim.getPrimitiveStart(p)
            e = prim.getPrimitiveEnd(p)

            color = None
            triPts = []
            for i in range(s, e):
                vi = prim.getVertex(i)
                vertex.setRow(vi)
                vertexColor.setRow(vi)
                v = vertex.getData3f()

                # NOTE: all vertex of the same polygon (triangles) should have the same color,
                #       so only grab it once.
                if color is None:
                    color = vertexColor.getData4f()
                    color = (color[0], color[1], color[2], color[3])

                triPts.append([v.x, v.y, v.z])
            triPts = np.array(triPts)

            # Apply transformation
            if transform is not None:
                v = transform.xformPoint(v)

            # calculate the semi-perimeter and area
            a = np.linalg.norm(triPts[0] - triPts[1], 2)
            b = np.linalg.norm(triPts[1] - triPts[2], 2)
            c = np.linalg.norm(triPts[2] - triPts[0], 2)
            s = (a + b + c) / 2
            area = (s * (s - a) * (s - b) * (s - c)) ** 0.5

            if color in colorsTotalAreas:
                colorsTotalAreas[color] += area
            else:
                colorsTotalAreas[color] = area

    areas = []
    rgbColors = []
    transparencies = []
    for color, area in six.iteritems(colorsTotalAreas):
        areas.append(area)
        rgbColors.append(list(color[:3]))

        # Check transparency
        isTransparent = color[3] < 1.0
        transparencies.append(isTransparent)

    return areas, rgbColors, transparencies


def getColorAttributesFromModel(model, region=None):
    # Calculate the net transformation
    transform = model.getNetTransform()
    transformMat = transform.getMat()

    areas = []
    rgbColors = []
    textures = []
    transparencies = []
    for nodePath in model.findAllMatches('**/+GeomNode'):
        geomNode = nodePath.node()
        if region is not None and region not in geomNode.getName():
            continue

        for n in range(geomNode.getNumGeoms()):
            state = geomNode.getGeomState(n)

            geom = geomNode.getGeom(n)

            area = getSurfaceAreaFromGeom(geom, transformMat)

            if state.hasAttrib(TextureAttrib.getClassType()):
                # Get color from texture
                texAttr = state.getAttrib(TextureAttrib.getClassType())
                tex = texAttr.getTexture()

                # Load texture image from file and compute average color
                texFilename = str(tex.getFullpath())
                img = imageio.imread(texFilename)

                texture = os.path.splitext(os.path.basename(texFilename))[0]

                assert img.dtype == np.uint8
                assert img.ndim == 3

                # TODO: handle black-and-white
                if img.shape[-1] == 3:
                    # RGB texture
                    rgbColor = (np.mean(img, axis=(0, 1)) / 255.0).tolist()
                elif img.shape[-1] == 4:
                    # RGBA texture
                    mask = img[:, :, -1] > 0.0
                    nbUnmasked = np.count_nonzero(mask)
                    rgbColor = (
                        1.0 / nbUnmasked * np.sum(img[:, :, :3] * mask[:, :, np.newaxis], axis=(0, 1)) / 255.0).tolist()
                else:
                    raise Exception('Unsupported image shape: %s' %
                                    (str(img.shape)))

                rgbColors.append(rgbColor)
                transparencies.append(False)
                areas.append(area)
                textures.append(texture)

            elif state.hasAttrib(ColorAttrib.getClassType()):
                colorAttr = state.getAttrib(ColorAttrib.getClassType())

                if (colorAttr.getColorType() == ColorAttrib.TFlat or colorAttr.getColorType() == ColorAttrib.TOff):
                    # Get flat color
                    color = colorAttr.getColor()

                    isTransparent = False
                    if isinstance(color, LVecBase4f):
                        rgbColor = [color[0], color[1], color[2]]
                        alpha = color[3]

                        if state.hasAttrib(TransparencyAttrib.getClassType()):
                            transAttr = state.getAttrib(
                                TransparencyAttrib.getClassType())
                            if transAttr.getMode() != TransparencyAttrib.MNone and alpha < 1.0:
                                isTransparent = True
                        elif alpha < 1.0:
                            isTransparent = True

                    elif isinstance(color, LVecBase3f):
                        rgbColor = [color[0], color[1], color[2]]
                    else:
                        raise Exception('Unsupported color class type: %s' % (
                            color.__class__.__name__))

                    rgbColors.append(rgbColor)
                    transparencies.append(isTransparent)
                    areas.append(area)
                    textures.append(None)

                else:
                    # Get colors from vertex data
                    verAreas, verRgbColors, vertransparencies = getColorAttributesFromVertexData(
                        geom, transformMat)
                    areas.extend(verAreas)
                    rgbColors.extend(verRgbColors)
                    transparencies.extend(vertransparencies)
                    textures.extend([None, ] * len(vertransparencies))
    areas = np.array(areas)
    areas /= np.sum(areas)

    return areas, rgbColors, transparencies, textures


class RgbRenderer1(object):
    def __init__(self, scene, size=(512, 512), mode='offscreen', zNear=0.1, zFar=1000.0, fov=40.0, cameraTransform=None):

        # Off-screen buffers are not supported in OSX
        if sys.platform == 'darwin':
            mode = 'onscreen'

        super(RgbRenderer1, self).__init__()

        self.__dict__.update(scene=scene, size=size, mode=mode, zNear=zNear, zFar=zFar, fov=fov,
                             cameraTransform=cameraTransform)

        self.cameraMask = BitMask32.bit(0)
        self.graphicsEngine = GraphicsEngine.getGlobalPtr()
        self.loader = Loader.getGlobalPtr()
        self.graphicsEngine.setDefaultLoader(self.loader)

        # Change some scene attributes for rendering
        self.scene.scene.setAttrib(RescaleNormalAttrib.makeDefault())
        self.scene.scene.setTwoSided(True)

        selection = GraphicsPipeSelection.getGlobalPtr()
        self.pipe = selection.makeDefaultPipe()
        logger.debug('Using %s' % (self.pipe.getInterfaceName()))

        # Attach a camera to every agent in the scene
        self.cameras = []
        for agentNp in self.scene.scene.findAllMatches('**/agents/agent*'):
            # a=agentNp.getChild(0).getChild(1) ## point it to the Neck
#            a=agentNp.getChild(0).getChild(0).getChild(1) # point to the eye

            a=agentNp

            camera = a.attachNewNode(ModelNode('camera-rgb'))
#            if self.cameraTransform is not None:
#                camera.setTransform(cameraTransform)

            camera.setPos(0,-0.5,2)
            camera.setHpr(180,0,0)
            camera.reparentTo(a)
            camera.node().setPreserveTransform(ModelNode.PTLocal)
            self.cameras.append(camera)

        self.rgbBuffers = dict()
        self.rgbTextures = dict()

        self._initRgbCapture()
        self._addDefaultLighting()

        self.notifySceneChanged()

    def notifySceneChanged(self):

        for modelNp in self.scene.scene.findAllMatches('**/model-*'):

            isInitialized = False
            objectNp = modelNp.getParent()
            for childNp in modelNp.getChildren():
                if childNp.getName() == 'render-rgb':
                    isInitialized = True
                    break

            if not isInitialized:
                rendererNp = objectNp.attachNewNode('render-rgb')
                model = modelNp.copyTo(rendererNp)

                # Set the model to be visible only to this camera
                model.hide(BitMask32.allOn())
                model.show(self.cameraMask)

                # Reparent render node below the existing physic node (if any)
                physicsNp = objectNp.find('**/physics')
                if not physicsNp.isEmpty():
                    rendererNp.reparentTo(physicsNp)

    def setBackgroundColor(self, rgba):
        for buf in six.itervalues(self.rgbBuffers):
            buf.setClearColor(LVector4f(*rgba))

    def _initRgbCapture(self):

        for camera in self.cameras:

            camNode = Camera('RGB camera')
            camNode.setCameraMask(self.cameraMask)
            lens = PerspectiveLens()
            lens.setFov(self.fov)
            lens.setAspectRatio(float(self.size[0]) / float(self.size[1]))
            lens.setNear(self.zNear)
            lens.setFar(self.zFar)
            camNode.setLens(lens)
            camNode.setScene(self.scene.scene)
            cam = camera.attachNewNode(camNode)

            winprops = WindowProperties.size(self.size[0], self.size[1])
            fbprops = FrameBufferProperties.getDefault()
            fbprops = FrameBufferProperties(fbprops)
            fbprops.setRgbaBits(8, 8, 8, 0)

            flags = GraphicsPipe.BFFbPropsOptional
            if self.mode == 'onscreen':
                flags = flags | GraphicsPipe.BFRequireWindow
            elif self.mode == 'offscreen':
                flags = flags | GraphicsPipe.BFRefuseWindow
            else:
                raise Exception('Unsupported rendering mode: %s' % (self.mode))

            buf = self.graphicsEngine.makeOutput(self.pipe, 'RGB buffer Rendering', 0, fbprops,
                                                 winprops, flags)
            if buf is None:
                raise Exception('Unable to create RGB buffer')

            # Set to render at the end
            buf.setSort(10000)

            dr = buf.makeDisplayRegion()
            dr.setSort(0)
            dr.setCamera(cam)
            dr = camNode.getDisplayRegion(0)

            tex = Texture()
            tex.setFormat(Texture.FRgb8)
            tex.setComponentType(Texture.TUnsignedByte)
            buf.addRenderTexture(
                tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)
            tex.makeRamImage()
            # XXX: should use tex.setMatchFramebufferFormat(True)?
            # agent=camera.getParent() # This is when camer is mounted on agent
            agent = camera.getParent()
            self.rgbBuffers[agent.getName()] = buf
            self.rgbTextures[agent.getName()] = tex

    def showRoomLayout(self, showCeilings=True, showWalls=True, showFloors=True):

        for np in self.scene.scene.findAllMatches('**/layouts/**/render-rgb/*c'):
            if showCeilings:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

        for np in self.scene.scene.findAllMatches('**/layouts/**/render-rgb/*w'):
            if showWalls:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

        for np in self.scene.scene.findAllMatches('**/layouts/**/render-rgb/*f'):
            if showFloors:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

    def destroy(self):
        self.graphicsEngine.removeAllWindows()
        del self.pipe

    def getRgbImage(self, agentId, channelOrder="RGB"):

        self.graphicsEngine.renderFrame()

        # NOTE: we need to call frame rendering twice in onscreen mode because
        # of double-buffering
        if self.mode == 'onscreen':
            self.graphicsEngine.renderFrame()
        tex = self.rgbTextures[agentId]

        # XXX: not sure about calling makeRamImage() before getting the image data, since it returns an empty image
        # and overwrite any previously rendered image. We may just call it
        # once when we create the texture.
        if not tex.mightHaveRamImage():
            tex.makeRamImage()

        if sys.version_info[0] < 3:
            data = tex.getRamImageAs(channelOrder).getData()   # Python 2
        else:
            # NOTE: see https://github.com/panda3d/panda3d/issues/173
            data = bytes(memoryview(
                tex.getRamImageAs(channelOrder)))  # Python 3

        # Must match Texture.TUnsignedByte
        image = np.frombuffer(data, dtype=np.uint8)

        image.shape = (tex.getYSize(), tex.getXSize(), len(channelOrder))
        image = np.flipud(image)

        return image

    def _addDefaultLighting(self):
        alight = AmbientLight('alight')
        alight.setColor(LVector4f(0.2, 0.2, 0.2, 1))
        alnp = self.scene.scene.attachNewNode(alight)
        self.scene.scene.setLight(alnp)

        for camera in self.cameras:

            # NOTE: Point light following the camera
            plight = PointLight('plight')
            plight.setColor(LVector4f(1.0, 1.0, 1.0, 1))
            plnp = camera.attachNewNode(plight)
            self.scene.scene.setLight(plnp)
