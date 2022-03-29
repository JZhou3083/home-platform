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

from __future__ import print_function

import os
import json
import re
import six
import random
import glob
import csv
import numpy as np
import subprocess
import logging

from panda3d.core import NodePath, Loader, LoaderOptions, Filename, TransformState,\
    LMatrix4f, Spotlight, LVector3f, PointLight, PerspectiveLens, CS_zup_right, CS_yup_right,\
    BitMask32, ModelNode

from home_platform.importer import obj2egg
from home_platform.constants import MODEL_CATEGORY_MAPPING
from home_platform.core import Scene
from home_platform.utils import mat4ToNumpyArray

logger = logging.getLogger(__name__)


def loadModel(modelPath):
    loader = Loader.getGlobalPtr()
    # NOTE: disable disk and RAM caching to avoid filling memory when loading
    # multiple scenes
    loaderOptions = LoaderOptions(LoaderOptions.LF_no_cache)
    node = loader.loadSync(Filename(modelPath), loaderOptions)
    if node is not None:
        nodePath = NodePath(node)
        nodePath.setTag('model-filename', os.path.abspath(modelPath))
    else:
        raise IOError('Could not load model file: %s' % (modelPath))
    return nodePath


def ignoreVariant(modelId):
    suffix = "_0"
    if modelId.endswith(suffix):
        modelId = modelId[:len(modelId) - len(suffix)]
    return modelId


def data_dir():
    """ Get SUNCG data path (must be symlinked to ~/.suncg)

    :return: Path to suncg dataset
    """

    if 'SUNCG_DATA_DIR' in os.environ:
        path = os.path.abspath(os.environ['SUNCG_DATA_DIR'])
    else:
        path = os.path.join(os.path.abspath(os.path.expanduser('~')), ".suncg")

    rooms_exist = os.path.isdir(os.path.join(path, "room"))
    houses_exist = os.path.isdir(os.path.join(path, "house"))
    if not os.path.isdir(path) or not rooms_exist or not houses_exist:
        raise Exception("Couldn't find the SUNCG dataset in '~/.suncg' or with environment variable SUNCG_DATA_DIR. "
                        "Please symlink the dataset there, so that the folders "
                        "'~/.suncg/room', '~/.suncg/house', etc. exist.")

    return path


def get_available_houses():
    path = data_dir()
    print("DBG: SUNCG path:", path)
    houses = os.listdir(os.path.join(path, "house"))
    return sorted(houses)


class ModelInformation(object):
    header = 'id,front,nmaterials,minPoint,maxPoint,aligned.dims,index,variantIds'

    def __init__(self, filename):
        self.model_info = {}

        self._parseFromCSV(filename)

    def _parseFromCSV(self, filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    rowStr = ','.join(row)
                    assert rowStr == ModelInformation.header
                else:
                    model_id, front, nmaterials, minPoint, maxPoint, \
                        aligned_dims, _, variantIds = row
                    if model_id in self.model_info:
                        raise Exception(
                            'Model %s already exists!' % (model_id))

                    front = np.fromstring(front, dtype=np.float64, sep=',')
                    nmaterials = int(nmaterials)
                    minPoint = np.fromstring(
                        minPoint, dtype=np.float64, sep=',')
                    maxPoint = np.fromstring(
                        maxPoint, dtype=np.float64, sep=',')
                    aligned_dims = np.fromstring(
                        aligned_dims, dtype=np.float64, sep=',')
                    variantIds = variantIds.split(',')
                    self.model_info[model_id] = {'model-id': model_id,
                                                 'front': front,
                                                 'nmaterials': nmaterials,
                                                 'minPoint': minPoint,
                                                 'maxPoint': maxPoint,
                                                 'aligned-dims': aligned_dims,
                                                 'variantIds': variantIds}

    def getModelInfo(self, modelId):
        return self.model_info[ignoreVariant(modelId)]


class ModelCategoryMapping(object):
    def __init__(self, filename):
        self.model_id = []
        self.fine_grained_class = {}
        self.coarse_grained_class = {}
        self.nyuv2_40class = {}
        self.wnsynsetid = {}
        self.wnsynsetkey = {}
        self.model_info = {}

        self._parseFromCSV(filename)

    def _parseFromCSV(self, filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    rowStr = ','.join(row)
                    assert rowStr == MODEL_CATEGORY_MAPPING["header"]
                else:
                    _, model_id, fine_grained_class, \
                        coarse_grained_class, _, nyuv2_40class, \
                        wnsynsetid, wnsynsetkey = row
                    if model_id in self.model_id:
                        raise Exception(
                            'Model %s already exists!' % (model_id))
                    self.model_id.append(model_id)

                    self.fine_grained_class[model_id] = fine_grained_class
                    self.coarse_grained_class[model_id] = coarse_grained_class
                    self.nyuv2_40class[model_id] = nyuv2_40class
                    self.wnsynsetid[model_id] = wnsynsetid
                    self.wnsynsetkey[model_id] = wnsynsetkey

                    self.model_info[model_id] = {'model-id': model_id,
                                                 'fine-class': fine_grained_class,
                                                 'coarse-class': coarse_grained_class,
                                                 'nyuv2-class': nyuv2_40class,
                                                 'wnsynsetid': wnsynsetid,
                                                 'wnsynsetkey': wnsynsetkey}

    def _printFineGrainedClassListAsDict(self):
        for c in sorted(set(self.fine_grained_class.values())):
            name = c.replace("_", " ")
            print("'%s':'%s'," % (c, name))

    def _printCoarseGrainedClassListAsDict(self):
        for c in sorted(set(self.coarse_grained_class.values())):
            name = c.replace("_", " ")
            print("'%s':'%s'," % (c, name))

    def getFineGrainedCategoryForModelId(self, modelId):
        return self.fine_grained_class[ignoreVariant(modelId)]

    def getCoarseGrainedCategoryForModelId(self, modelId):
        return self.coarse_grained_class[ignoreVariant(modelId)]

    def getFineGrainedClassList(self):
        return sorted(set(self.fine_grained_class.values()))

    def getCoarseGrainedClassList(self):
        return sorted(set(self.coarse_grained_class.values()))


class ObjectVoxelData(object):
    def __init__(self, voxels, translation, scale):
        self.voxels = voxels
        self.translation = translation
        self.scale = scale

    def getFilledVolume(self):
        nbFilledVoxels = np.count_nonzero(self.voxels)
        perVoxelVolume = self.scale / np.prod(self.voxels.shape)
        return nbFilledVoxels * perVoxelVolume

    @staticmethod
    def fromFile(filename):

        with open(filename, 'rb') as f:

            # Read header line and version
            line = f.readline().decode('ascii').strip()  # u'#binvox 1'
            header, version = line.split(" ")
            if header != '#binvox':
                raise Exception(
                    'Unable to read header from file: %s' % (filename))
            version = int(version)
            assert version == 1

            # Read dimensions and transforms
            line = f.readline().decode('ascii').strip()  # u'dim 128 128 128'
            items = line.split(" ")
            assert items[0] == 'dim'
            depth, height, width = np.fromstring(
                " ".join(items[1:]), sep=' ', dtype=np.int)

            # XXX: what is this translation component?
            # u'translate -0.176343 -0.356254 0.000702'
            line = f.readline().decode('ascii').strip()
            items = line.split(" ")
            assert items[0] == 'translate'
            translation = np.fromstring(
                " ".join(items[1:]), sep=' ', dtype=np.float)

            line = f.readline().decode('ascii').strip()  # u'scale 0.863783'
            items = line.split(" ")
            assert items[0] == 'scale'
            scale = float(items[1])

            # Read voxel data
            line = f.readline().decode('ascii').strip()  # u'data'
            assert line == 'data'

            size = width * height * depth
            voxels = np.zeros((size,), dtype=np.int8)

            nrVoxels = 0
            index = 0
            endIndex = 0
            while endIndex < size:
                value = np.frombuffer(f.read(1), dtype=np.uint8)[0]
                count = np.frombuffer(f.read(1), dtype=np.uint8)[0]
                endIndex = index + count
                assert endIndex <= size

                voxels[index:endIndex] = value
                if value != 0:
                    nrVoxels += count
                index = endIndex

            # NOTE: we should by now have reach the end of the file
            assert len(f.readline()) == 0

            # FIXME: not sure about the particular dimension ordering here!
            voxels = voxels.reshape((width, height, depth))

            logger.debug(
                'Number of non-empty voxels read from file: %d' % (nrVoxels))

        return ObjectVoxelData(voxels, translation, scale)


def reglob(path, exp):
    # NOTE: adapted from https://stackoverflow.com/questions/13031989/regular-expression-using-in-glob-glob-of-python
    # Example: "fr_0rm_3[a-z].bam" To ['/home/jzhou3083/work/home-platform/tests/home_platform/../data/suncg/room/0004d52d1aeeb8ae6de39d6bd993e992/fr_0rm_3f.bam',...]
    # Have the paths to all the layout files of the room stored in the list variable res
    m = re.compile(exp)
    res = [f for f in os.listdir(path) if m.search(f)]
    res = map(lambda x: "%s/%s" % (path, x, ), res)
    return res


class SunCgModelLights(object):

    def __init__(self, filename):

        with open(filename) as f:
            self.data = json.load(f)

        self.supportedModelIds = self.data.keys()

    def getLightsForModel(self, modelId):
        lights = []
        if modelId in self.supportedModelIds:

            for n, lightData in enumerate(self.data[modelId]):

                attenuation = LVector3f(*lightData['attenuation'])

                # TODO: implement light power
                # power = float(lightData['power'])

                positionYup = LVector3f(*lightData['position'])
                yupTozupMat = LMatrix4f.convertMat(CS_yup_right, CS_zup_right)
                position = yupTozupMat.xformVec(positionYup)

                colorHtml = lightData['color']
                color = LVector3f(*[int('0x' + colorHtml[i:i + 2], 16)
                                    for i in range(1, len(colorHtml), 2)]) / 255.0

                direction = None
                lightType = lightData['type']
                lightName = modelId + '-light-' + str(n)
                if lightType == 'SpotLight':
                    light = Spotlight(lightName)
                    light.setAttenuation(attenuation)
                    light.setColor(color)

                    cutoffAngle = float(lightData['cutoffAngle'])
                    lens = PerspectiveLens()
                    lens.setFov(cutoffAngle / np.pi * 180.0)
                    light.setLens(lens)

                    # NOTE: unused attributes
                    # dropoffRate = float(lightData['dropoffRate'])

                    directionYup = LVector3f(*lightData['direction'])
                    direction = yupTozupMat.xformVec(directionYup)

                elif lightType == 'PointLight':
                    light = PointLight(lightName)
                    light.setAttenuation(attenuation)
                    light.setColor(color)

                elif lightType == 'LineLight':
                    # XXX: we may wish to use RectangleLight from the devel
                    # branch of Panda3D
                    light = PointLight(lightName)
                    light.setAttenuation(attenuation)
                    light.setColor(color)

                    # NOTE: unused attributes
                    # dropoffRate = float(lightData['dropoffRate'])
                    # cutoffAngle = float(lightData['cutoffAngle'])

                    # position2Yup = LVector3f(*lightData['position2'])
                    # position2 = yupTozupMat.xformVec(position2Yup)

                    # directionYup = LVector3f(*lightData['direction'])
                    # direction = yupTozupMat.xformVec(directionYup)

                else:
                    raise Exception('Unsupported light type: %s' % (lightType))

                lightNp = NodePath(light)

                # Set position and direction of light
                lightNp.setPos(position)
                if direction is not None:
                    targetPos = position + direction
                    lightNp.look_at(targetPos, LVector3f.up())

                lights.append(lightNp)

        return lights

    def isModelSupported(self, modelId):
        isSupported = False
        if modelId in self.supportedModelIds:
            isSupported = True
        return isSupported


def subdiviseLayoutObject(layoutNp):
    objectModelId = layoutNp.getTag('model-id')

    geomNodes = list(layoutNp.findAllMatches('**/+GeomNode'))

    layoutNps = []
    if objectModelId.endswith('w'):
        # Wall
        # Regroup WallInside and WallOutside geom nodes
        maxIdx = np.max([int(geomNodes[i].getName().split('_')[-1])
                         for i in range(len(geomNodes))])
        wallGeomNodes = [[] for _ in range(maxIdx + 1)]
        for i in range(len(geomNodes)):
            name = geomNodes[i].getName()
            idx = int(name.split('_')[-1])
            wallGeomNodes[idx].append(geomNodes[i])

        for i in range(len(wallGeomNodes)):

            instanceId = str(objectModelId) + '-' + str(i)
            objectNp = NodePath('object-' + instanceId)
            objectNp.setTag('model-id', objectModelId)
            objectNp.setTag('instance-id', instanceId)

            model = NodePath(ModelNode('model-' + instanceId))
            # model.setTag('model-filename', os.path.abspath(modelPath))
            model.reparentTo(objectNp)
            model.hide(BitMask32.allOn())

            for geomNode in wallGeomNodes[i]:
                geomNode.reparentTo(model)

            layoutNps.append(objectNp)
    else:
        # Floor or ceiling
        layoutNps.append(layoutNp)

    return layoutNps


class SunCgSceneLoader(object):

    @staticmethod
    def getHouseJsonPath(base_path, house_id):
        return os.path.join(
            base_path,
            "house",
            house_id,
            "house.json")

    @staticmethod
    def loadHouseFromJson(houseId, datasetRoot):

        filename = SunCgSceneLoader.getHouseJsonPath(datasetRoot, houseId)
        with open(filename) as f:
            data = json.load(f)
        assert houseId == data['id']
        houseId = str(data['id'])

        # Create new node for house instance
        houseNp = NodePath('house-' + str(houseId))
        houseNp.setTag('house-id', str(houseId))

        objectIds = {}
        for levelId, level in enumerate(data['levels']):
            logger.debug('Loading Level %s to scene' % (str(levelId)))

            # Create new node for level instance
            levelNp = houseNp.attachNewNode('level-' + str(levelId))
            levelNp.setTag('level-id', str(levelId))

            levelObjectsNp = levelNp.attachNewNode('objects')

            roomNpByNodeIndex = {}
            for nodeIndex, node in enumerate(level['nodes']):

                if not node['valid'] == 1:
                    continue

                if node['type'] == 'Box':
                    pass
                elif node['type'] == 'Room':
                    modelId = str(node['modelId'])
                    logger.debug('Loading Room %s to scene' % (modelId))

                    # Create new nodes for room instance
                    instanceId = str(modelId) + '-0'
                    roomNp = levelNp.attachNewNode('room-' + instanceId)
                    roomNp.setTag('model-id', modelId)
                    roomNp.setTag('instance-id', instanceId)
                    roomNp.setTag('room-id', instanceId)

                    roomLayoutsNp = roomNp.attachNewNode('layouts')
                    roomObjectsNp = roomNp.attachNewNode('objects')

                    # Include some semantic information
                    roomTypes = []
                    for roomType in node['roomTypes']:
                        roomType = roomType.lower().strip()
                        if len(roomType) > 0:
                            roomTypes.append(roomType)
                    roomNp.setTag('room-types', ','.join(roomTypes))

                    # Load models defined for this room
                    for roomObjFilename in reglob(os.path.join(datasetRoot, 'room', houseId),
                                                  modelId + '[a-z].bam'):

                        # NOTE: loading the BAM format is faster and more efficient
                        # Convert extension from OBJ + MTL to BAM format
                        f, _ = os.path.splitext(roomObjFilename)
                        bamModelFilename = f + ".bam"
                        eggModelFilename = f + ".egg"
                        if os.path.exists(bamModelFilename):
                            modelFilename = bamModelFilename
                        elif os.path.exists(eggModelFilename):
                            modelFilename = eggModelFilename
                        else:
                            raise Exception(
                                'The SUNCG dataset object models need to be convert to Panda3D EGG or BAM format!')

                        # Create new node for object instance
                        objectModelId = os.path.splitext(
                            os.path.basename(roomObjFilename))[0]

                        model = loadModel(modelFilename)
                        instanceId = str(objectModelId) + '-0'
                        objectNp = NodePath('object-' + instanceId)
                        objectNp.setTag('model-id', objectModelId)
                        objectNp.setTag('instance-id', instanceId)

                        model.setName('model-' + os.path.basename(f))
                        model.reparentTo(objectNp)
                        model.hide(BitMask32.allOn())

                        for subObjectNp in subdiviseLayoutObject(objectNp):
                            subObjectNp.reparentTo(roomLayoutsNp)


                    if 'nodeIndices' in node:
                        for childNodeIndex in node['nodeIndices']:
                            roomNpByNodeIndex[childNodeIndex] = roomObjectsNp
                elif node['type'] == 'Object':
                    modelId = str(node['modelId'])
                    logger.debug('Loading Object %s to scene' % (modelId))

                    # Instance identification
                    if modelId in objectIds:
                        objectIds[modelId] = objectIds[modelId] + 1
                    else:
                        objectIds[modelId] = 0

                    # Create new node for object instance
                    instanceId = str(modelId) + '-' + str(objectIds[modelId])
                    objectNp = NodePath('object-' + instanceId)
                    objectNp.setTag('model-id', modelId)
                    objectNp.setTag('instance-id', instanceId)

                    # NOTE: loading the BAM format is faster and more efficient
                    # Convert extension from OBJ + MTL to BAM format
                    objFilename = os.path.join(
                        datasetRoot, 'object', node['modelId'], node['modelId'] + '.bam')
                    assert os.path.exists(objFilename)
                    f, _ = os.path.splitext(objFilename)

                    bamModelFilename = f + ".bam"
                    eggModelFilename = f + ".egg"
                    if os.path.exists(bamModelFilename):
                        modelFilename = bamModelFilename
                    elif os.path.exists(eggModelFilename):
                        modelFilename = eggModelFilename
                    else:
                        raise Exception(
                            'The SUNCG dataset object models need to be convert to Panda3D EGG or BAM format!')

                    model = loadModel(modelFilename)
                    model.setName('model-' + os.path.basename(f))
                    model.reparentTo(objectNp)
                    model.hide(BitMask32.allOn())

                    # 4x4 column-major transformation matrix from object
                    # coordinates to scene coordinates
                    transform = np.array(node['transform']).reshape((4, 4))

                    # Transform from Y-UP to Z-UP coordinate systems
                    # TODO: use Mat4.convertMat(CS_zup_right, CS_yup_right)
                    yupTransform = np.array([[1, 0, 0, 0],
                                             [0, 0, -1, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 0, 1]])

                    zupTransform = np.array([[1, 0, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, -1, 0, 0],
                                             [0, 0, 0, 1]])

                    transform = np.dot(
                        np.dot(yupTransform, transform), zupTransform)
                    transform = TransformState.makeMat(
                        LMatrix4f(*transform.ravel()))

                    # Calculate the center of this object
                    minBounds, maxBounds = model.getTightBounds()
                    centerPos = minBounds + (maxBounds - minBounds) / 2.0

                    # Add offset transform to make position relative to the
                    # center
                    objectNp.setTransform(transform.compose(
                        TransformState.makePos(centerPos)))
                    model.setTransform(TransformState.makePos(-centerPos))

                    # Get the parent nodepath for the object (room or level)
                    if nodeIndex in roomNpByNodeIndex:
                        objectNp.reparentTo(roomNpByNodeIndex[nodeIndex])
                    else:
                        objectNp.reparentTo(levelObjectsNp)

                    # Validation
                    assert np.allclose(mat4ToNumpyArray(model.getNetTransform().getMat()),
                                       mat4ToNumpyArray(transform.getMat()), atol=1e-6)

                elif node['type'] == 'Ground':
                    modelId = str(node['modelId'])
                    logger.debug('Loading Ground %s to scene' % (modelId))

                    # Create new nodes for ground instance
                    instanceId = str(modelId) + '-0'
                    groundNp = levelNp.attachNewNode('ground-' + instanceId)
                    groundNp.setTag('instance-id', instanceId)
                    groundNp.setTag('model-id', modelId)
                    groundNp.setTag('ground-id', instanceId)
                    groundLayoutsNp = groundNp.attachNewNode('layouts')

                    # Load model defined for this ground
                    for groundObjFilename in reglob(os.path.join(datasetRoot, 'room', houseId),
                                                    modelId + '[a-z].bam'):

                        # NOTE: loading the BAM format is faster and more efficient
                        # Convert extension from OBJ + MTL to BAM format
                        f, _ = os.path.splitext(groundObjFilename)
                        bamModelFilename = f + ".bam"
                        eggModelFilename = f + ".egg"
                        if os.path.exists(bamModelFilename):
                            modelFilename = bamModelFilename
                        elif os.path.exists(eggModelFilename):
                            modelFilename = eggModelFilename
                        else:
                            raise Exception(
                                'The SUNCG dataset object models need to be convert to Panda3D EGG or BAM format!')

                        objectModelId = os.path.splitext(
                            os.path.basename(groundObjFilename))[0]
                        instanceId = str(objectModelId) + '-0'
                        objectNp = NodePath('object-' + instanceId)
                        objectNp.reparentTo(groundLayoutsNp)
                        objectNp.setTag('model-id', objectModelId)
                        objectNp.setTag('instance-id', instanceId)

                        model = loadModel(modelFilename)
                        model.setName('model-' + os.path.basename(f))
                        model.reparentTo(objectNp)
                        model.hide(BitMask32.allOn())

                else:
                    raise Exception('Unsupported node type: %s' %
                                    (node['type']))

        scene = Scene('house-' + houseId)
        houseNp.reparentTo(scene.scene)

        # Recenter objects in rooms
        for room in scene.scene.findAllMatches('**/room*'):

            # Calculate the center of this room
            bounds = room.getTightBounds()
            if bounds is not None:
                minBounds, maxBounds = room.getTightBounds()
                centerPos = minBounds + (maxBounds - minBounds) / 2.0

                # Add offset transform to room node
                room.setTransform(TransformState.makePos(centerPos))

                # Add recentering transform to all children nodes
                for childNp in room.getChildren():
                    childNp.setTransform(TransformState.makePos(-centerPos))
            else:
                # This usually means the room has no layout or objects
                pass

        # Recenter objects in grounds
        for ground in scene.scene.findAllMatches('**/ground*'):

            # Calculate the center of this ground
            minBounds, maxBounds = ground.getTightBounds()
            centerPos = minBounds + (maxBounds - minBounds) / 2.0

            # Add offset transform to ground node
            ground.setTransform(TransformState.makePos(centerPos))

            # Add recentering transform to all children nodes
            for childNp in ground.getChildren():
                childNp.setTransform(TransformState.makePos(-centerPos))

        return scene


class HouseScoreInformation(object):
    header = 'sceneId,level,posvote,negvote'

    def __init__(self, filename):
        self.house_info = {}

        self._parseFromCSV(filename)

    def _parseFromCSV(self, filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    rowStr = ','.join(row)
                    assert rowStr == HouseScoreInformation.header
                else:
                    houseId, level, posvote, negvote = row
                    level = int(level)
                    posvote = int(posvote)
                    negvote = int(negvote)

                    levelScore = float(posvote) / (posvote + negvote)

                    levels = []
                    if houseId in self.house_info:
                        levels = self.house_info[houseId]
                    levels.append((level, levelScore))

                    self.house_info[houseId] = levels

    def getHouseScore(self, houseId):
        try:
            levels = self.house_info[houseId]
            score = np.mean([levelScore for _, levelScore in levels])
        except KeyError:
            score = 0.0
        return score


def convert3dModelsToPanda3d(datasetRoot, houseId, eggFormat=True, bamFormat=True, overwrite=False):

    for objFilename in glob.iglob(os.path.join(datasetRoot, 'room/%s/*.obj' % (houseId))):

        cwd = os.path.dirname(objFilename)

        objFilenameNoExt, _ = os.path.splitext(objFilename)

        if eggFormat:
            eggFilename = objFilenameNoExt + '.egg'
            if not os.path.exists(eggFilename) or overwrite:
                obj2egg(objFilename, eggFilename,
                        coordinateSystem='y-up-right')
                if not os.path.exists(eggFilename):
                    logger.warning(
                        "Could not find output file %s. An error probably occured during conversion." % (eggFilename))

        if bamFormat:
            bamFilename = objFilenameNoExt + '.bam'
            if not os.path.exists(bamFilename) or overwrite:
                # FIXME: the current working directory seems to get messed up when we can the function directly
                # obj2bam(objFilename, bamFilename,
                #         coordinateSystem='y-up-right')
                subprocess.call('egg2bam -ps rel -o %s %s' %
                                (bamFilename, eggFilename), cwd=cwd, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if not os.path.exists(bamFilename):
                    logger.warning(
                        "Could not find output file %s. An error probably occured during conversion." % (bamFilename))


def getSurfaceAreaFromBoundingBox(minBoxPt, maxBoxPt):
    length = maxBoxPt[0] - minBoxPt[0]
    width = maxBoxPt[1] - minBoxPt[1]
    area = length * width
    return area


def filterBlacklist(houseIds, filename):

    # Read selection from JSON file
    with open(filename, 'r') as f:
        data = json.load(f)

    # Get the full list of blacklisted house ids
    blacklistedHouseIds = sum([v for v in six.itervalues(data)], [])

    filteredHouseIds = [
        houseId for houseId in houseIds if houseId not in blacklistedHouseIds]

    nbFiltered = len(houseIds) - len(filteredHouseIds)
    percentageFiltered = float(nbFiltered) / len(houseIds) * 100.0
    logger.info(
        'Filtered %d houses (%4.2f%%) that have been blacklisted' % (nbFiltered, percentageFiltered))

    return filteredHouseIds


def filterRealisticSceneLayout(datasetRoot, houseIds):
    """
    NOTE: same criteria as in
      A. Das, S. Datta, G. Gkioxari, S. Lee, D. Parikh, and D. Batra, "Embodied Question Answering," 2017.
    """
    filename = os.path.join(datasetRoot, 'metadata', 'houseAnnoMturk.csv')
    houseInfo = HouseScoreInformation(filename)

    filteredHouseIds = []
    for i, houseId in enumerate(houseIds):
        score = houseInfo.getHouseScore(houseId)

        # Must have been judged realistic (perfect score for all levels)
        if score == 1.0:
            filteredHouseIds.append(houseId)

        if (i + 1) % 10000 == 0:
            logger.debug('Processed %d total houses (%d valid)' %
                         (i + 1, len(filteredHouseIds)))

    nbFiltered = len(houseIds) - len(filteredHouseIds)
    percentageFiltered = float(nbFiltered) / len(houseIds) * 100.0
    logger.info(
        'Filtered %d houses (%4.2f%%) that do not have a realistic scene layout' % (nbFiltered, percentageFiltered))

    return filteredHouseIds


def splitTrainValidTest(ids, trainRatio=0.7, validRatio=0.2, testRatio=0.1):

    # Randomize order
    random.shuffle(ids)

    assert (trainRatio + validRatio + testRatio) <= 1.0
    nbTrainSamples = int(trainRatio * len(ids))
    nbValidSamples = int(validRatio * len(ids))
    nbTestSamples = len(ids) - (nbTrainSamples + nbValidSamples)

    trainIds = ids[:nbTrainSamples]
    valIds = ids[nbTrainSamples:nbTrainSamples + nbValidSamples]
    testIds = ids[-nbTestSamples:]
    assert len(trainIds) + len(valIds) + len(testIds) == len(ids)

    return trainIds, valIds, testIds
