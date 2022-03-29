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
import sys
import scipy.io
import scipy.signal
import logging
import soundfile as sf
from scipy.io import wavfile
import numpy as np
import math
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from pyroomacoustics.soundsource import SoundSource
from six import itervalues, iteritems
from scipy import signal
from string import digits
# from evert import Room as EvertRoom
from evert import Source, Listener, Vector3, Matrix3, Polygon, PathSolution
from panda3d.core import NodePath, LVector3f, LVecBase3, Material, TransformState, AudioSound, CS_zup_right, BitMask32, ClockObject
from direct.task.TaskManagerGlobal import taskMgr

from home_platform.core import World
from home_platform.realRoom import loadModel
from home_platform.rendering import get3DTrianglesFromModel, getColorAttributesFromModel
from home_platform.utils import vec3ToNumpyArray

logger = logging.getLogger(__name__)

MODEL_DATA_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "data", "models")

def getMaterial(model):
    _, _, _, textures = getColorAttributesFromModel(model)
    for texture in textures:
        if texture is None:
                texture = 'default'
        else:
            # Remove any digits
            if (sys.version_info > (3, 0)):
                texture = texture.translate(
                        {ord(c): '' for c in "1234567890"})
            else:
                texture = texture.translate(None, digits)

            # Remove trailing underscores
            texture = texture.split("_")[-1]

            # NOTE: handle many variations of textile and wood in SUNCG
            # texture names
            if "textile" in texture:
                texture = "textile"
            if "wood" in texture:
                texture = "wood"
        if texture not in TextureAbsorptionTable.textures:
            logger.warn(
                'Unsupported texture basename for material acoustics: %s' % (texture))
            texture = 'default'
        category, material = TextureAbsorptionTable.textures[texture]
    return material
def getAcousticPolysFromModel(model):
    Polys = []
    triangles = get3DTrianglesFromModel(model)
    materialName = getMaterial(model)
    material = pra.Material(energy_absorption=materialName, scattering=materialName)
    for triangle in triangles:
        triangle = triangle.T     ### pyroomacoustics store the coordinations differently from EVERT library
        pts = []
        for pt in triangle:
            pts.append(pt)
        Polys.append(pra.wall_factory(
                np.vstack(pts),
                material.energy_absorption["coeffs"],
                material.scattering["coeffs"],
            ))
    return Polys, triangles

def absorptCoeff(freqs,temp,humid,P_atm= 101.325, unit="dB"):
    P0 = 101.325
    T0 = 293.15
    temp = temp+273.15

    ratio_P = P_atm/P0
    ratio_temp = temp/T0

    #This part convert the relative humidity to molar conentration of water vapor
    C =-6.8346 * (273.16/temp)**1.261+4.6151
    P_sat = P0*10**C
    h = humid*(P_sat/P0)*ratio_P

    # relaxation frequencies
    f_ro = ratio_P*(4.04*10**(4)*h*(0.02+h)/(0.391+h)+24)

    f_rn= ratio_P*ratio_temp**(-0.5)*(280*h*np.exp(-4.170*(ratio_temp**(-1/3)-1))+9)


    alphas =[]
    for freq in freqs:
        alpha = 1.84*10**(-11)*ratio_P**(-1)*ratio_temp**(0.5)
        a= 0.01275*np.exp(-2239.1/temp)*f_ro/(f_ro**2+freq**2)
        a+= 0.1068*np.exp(-3352.0/temp)*f_rn/(f_rn**2+freq**2)
        alpha += ratio_temp**(-5/2)*a
        alpha *= 8.686*freq**2

        if unit == "ratio":
            alpha = (10**(alpha/20)-1)*1000
        alphas.append(alpha)

    return alphas
class HRTF(object):

    def __init__(self, nbChannels, samplingRate, maxLength=None):
        self.nbChannels = nbChannels
        self.samplingRate = samplingRate
        self.maxLength = maxLength
        self.elevations = None
        self.azimuts = None
        self.impulses = None
        self.channels = None

        self.timeMargin = 0
        self.impulsesFourier = None

    def _precomputeImpulsesFourier(self):

        N = self.impulses.shape[-1]
        if self.maxLength is not None:
            N = self.maxLength

        self.timeMargin = N
        self.impulsesFourier = np.fft.fft(self.impulses, N + self.timeMargin)

    def resample(self, newSamplingRate, maxLength=None):

        if maxLength is not None:
            self.maxLength = maxLength

        N = self.impulses.shape[-1]
        nbSamples = int(N * newSamplingRate / self.samplingRate)
        if nbSamples != N:
            # TODO: resampy function doesn't seem to work for 3D and 4D tensors (it returns only zeros)
            #             try:
            #                 # Use high quality resampling if available
            #                 # https://pypi.python.org/pypi/resampy
            #                 import resampy
            #                 self.impulses = resampy.resample(self.impulses, self.samplingRate, newSamplingRate, axis=-1)
            #             except ImportError:
            #                 logger.warn("Using lower quality resampling routine!")

            # TODO: for high-quality resampling, we may simply do linear
            # interpolation (but it is slower)
            self.impulses = scipy.signal.resample(
                self.impulses, nbSamples, axis=-1)

        self._precomputeImpulsesFourier()

    def getImpulseResponse(self, azimut, elevation):
        closestAzimutIdx = np.argmin(np.sqrt((self.azimuts - azimut) ** 2))
        closestElevationIdx = np.argmin(
            np.sqrt((self.elevations - elevation) ** 2))
        return self.impulses[closestAzimutIdx, closestElevationIdx]

    def getFourierImpulseResponse(self, azimut, elevation):
        if self.impulsesFourier is None:
            self._calculateImpulsesFourier()

        closestAzimutIdx = np.argmin(np.sqrt((self.azimuts - azimut) ** 2))
        closestElevationIdx = np.argmin(
            np.sqrt((self.elevations - elevation) ** 2))
        return self.impulsesFourier[closestAzimutIdx, closestElevationIdx]


def interauralPolarToVerticalPolarCoordinates(elevations, azimuts):

    elevations = np.atleast_1d(elevations)
    azimuts = np.atleast_1d(azimuts)

    # Convert interaural-polar coordinates to 3D cartesian coordinates on the
    # unit sphere
    x = np.cos(azimuts * np.pi / 180.0) * np.cos(elevations * np.pi / 180.0)
    y = np.sin(azimuts * np.pi / 180.0) * -1.0
    z = np.cos(azimuts * np.pi / 180.0) * np.sin(elevations * np.pi / 180.0)
    assert np.allclose(x**2 + y**2 + z**2, np.ones_like(elevations))

    # Convert 3D cartesian coordinates on the unit sphere to vertical-polar
    # coordinates
    azimuts = np.arctan2(-y, x) * 180.0 / np.pi
    elevations = np.arcsin(z) * 180.0 / np.pi

    return elevations, azimuts


def verticalPolarToInterauralPolarCoordinates(elevation, azimut):

    # Convert vertical-polar coordinates to 3D cartesian coordinates on the
    # unit sphere
    x = np.cos(elevation * np.pi / 180.0) * np.sin(azimut * np.pi / 180.0)
    y = np.cos(elevation * np.pi / 180.0) * np.cos(azimut * np.pi / 180.0)
    z = np.sin(elevation * np.pi / 180.0) * 1.0
    assert np.allclose(x**2 + y**2 + z**2, np.ones_like(elevation))

    # Convert 3D cartesian coordinates on the unit sphere to interaural-polar
    # coordinates
    azimut = np.arcsin(x) * 180.0 / np.pi
    elevation = np.arctan2(z, y) * 180.0 / np.pi

    return elevation, azimut


def verticalPolarToCipicCoordinates(elevation, azimut):

    elevation, azimut = verticalPolarToInterauralPolarCoordinates(
        elevation, azimut)

    # Find the principal value of an elevation, converting it to the range [90, 270]
    # See:
    # http://earlab.bu.edu/databases/collections/cipic/documentation/hrir_data_documentation.pdf
    elevation = np.arctan2(np.sin(elevation * np.pi / 180),
                           np.cos(elevation * np.pi / 180)) * 180.0 / np.pi

    if isinstance(elevation, np.ndarray):
        elevation[elevation < -90.0] += 360.0
    else:
        if elevation < -90:
            elevation += 360.0

    return elevation, azimut


class CipicHRTF(HRTF):
    def __init__(self, filename, samplingRate):

        super(CipicHRTF, self).__init__(nbChannels=2,
                                        samplingRate=44100.0)

        self.filename = filename

        # NOTE: CIPIC defines elevation and azimut angle in a interaural-polar coordinate system.
        #       We actually want to use the vertical-polar coordinate system.
        # see:
        # http://www.ece.ucdavis.edu/cipic/spatial-sound/tutorial/psychoacoustics-of-spatial-hearing/
        self.elevations = np.linspace(-45, 230.625, num=50) * np.pi / 180
        self.azimuts = np.concatenate(
            ([-80, -65, -55], np.linspace(-45, 45, num=19), [55, 65, 80])) * np.pi / 180

        self.impulses = self._loadImpulsesFromFile()
        self.channels = ['left', 'right']

        self.resample(samplingRate)
        self._precomputeImpulsesFourier()

    def _loadImpulsesFromFile(self):

        # Load CIPIC HRTF data
        cipic = scipy.io.loadmat(self.filename)
        hrirLeft = np.transpose(cipic['hrir_l'], [2, 0, 1])
        hrirRight = np.transpose(cipic['hrir_r'], [2, 0, 1])

        # Store impulse responses in time domain
        N = len(hrirLeft[:, 0, 0])
        impulses = np.zeros((len(self.azimuts), len(
            self.elevations), self.nbChannels, N))
        for i in range(len(self.azimuts)):
            for j in range(len(self.elevations)):
                impulses[i, j, 0, :] = hrirLeft[:, i, j]
                impulses[i, j, 1, :] = hrirRight[:, i, j]

        return impulses

    def getImpulseResponse(self, azimut, elevation):
        elevation, azimut = verticalPolarToCipicCoordinates(elevation, azimut)
        return super(CipicHRTF, self).getImpulseResponse(azimut, elevation)

    def getFourierImpulseResponse(self, azimut, elevation):
        elevation, azimut = verticalPolarToCipicCoordinates(elevation, azimut)
        return super(CipicHRTF, self).getFourierImpulseResponse(azimut, elevation)


class TextureAbsorptionTable(object):
    textures = {
        ""
        
        "default": ['hard surfaces', 'mat_CR2_concrete'],
        "floor": ['floor', 'mat_CR2_floor'],
        "glazing": ['glazing', 'mat_CR2_windows'],
        'ceiling': ['ceiling', 'mat_CR2_ceiling'],
        'concrete': ['hard surfaces', 'mat_CR2_concrete'],
        'plaster': ['coverings', 'mat_CR2_plaster']

    }

    @staticmethod
    def getMeanAbsorptionCoefficientsFromModel(model, units='dB'):

        # Get the list of materials
        areas, _, _, textures = getColorAttributesFromModel(model)
        totalCoefficients = np.zeros(len(MaterialAbsorptionTable.frequencies))

        for area, texture in zip(areas, textures):
            if texture is None:
                texture = 'default'
            else:
                # Remove any digits
                if (sys.version_info > (3, 0)):
                    texture = texture.translate(
                        {ord(c): '' for c in "1234567890"})
                else:
                    texture = texture.translate(None, digits)

                # Remove trailing underscores
                texture = texture.split("_")[-1]

                # NOTE: handle many variations of textile and wood in SUNCG
                # texture names
                if "textile" in texture:
                    texture = "textile"
                if "wood" in texture:
                    texture = "wood"
            if texture not in TextureAbsorptionTable.textures:
                logger.warn(
                    'Unsupported texture basename for material acoustics: %s' % (texture))
                texture = 'default'
            category, material = TextureAbsorptionTable.textures[texture]

            coefficients, _ = MaterialAbsorptionTable.getAbsorptionCoefficients(
                category, material, units='normalized')
            totalCoefficients += area * coefficients

        if units == 'dB':
            eps = np.finfo(np.float).eps
            totalCoefficients = 20.0 * np.log10(1.0 - coefficients + eps)
        elif units == 'normalized':
            # Nothing to do
            pass
        else:
            raise Exception('Unsupported units: %s' % (units))

        return totalCoefficients
class MaterialAbsorptionTable(object):
    # From: Auralization : fundamentals of acoustics, modelling, simulation, algorithms and acoustic virtual reality
    # https://cds.cern.ch/record/1251519/files/978-3-540-48830-9_BookBackMatter.pdf

    categories = ['hard surfaces', 'glazing','ceiling', 'floor','coverings']
    frequencies = [31.5, 63, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000]
    materials = [[  # hard surfaces, such as concrete,
        "mat_cr2_concrete",
    ],
        [  # Glazing
            "mat_cr2_windows",  # Glass window
    ],
        [  # ceiling
            "mat_cr2_ceiling",  # Ceiling
    ],
        [  # Floor
            "mat_cr2_floor",
    ],
        [  # coverings, such as plaster and curtains
            "mat_cr2_plaster"
    ],
    ]

    # Tables of random-incidence absorption coefficients
    table = [[  # hard surface
        [0.053, 0.077, 0.085, 0.075, 0.056,  0.059,  0.059, 0.044, 0.041, 0.037], # mat_CR2_concrete

    ],
        [  # glazing
            [0.263, 0.223, 0.175, 0.073, 0.049, 0.057, 0.133, 0.055, 0.053, 0.037], # mat_CR2_windows
    ],
        [  # ceiling
            [0.068, 0.007, 0.083, 0.104, 0.048, 0.049, 0.047, 0.062, 0.050, 0.049], # mat_CR2_ceiling
    ],
        [  # Floor
            [0.070, 0.086, 0.071, 0.091, 0.070, 0.065, 0.062, 0.043, 0.033, 0.029],
    ],
        [  # coverings
            [0.073, 0.075, 0.033, 0.050,  0.039, 0.044, 0.048, 0.036, 0.028, 0.036], # mat_CR2_plaster
    ],
        [  # Wall absorbers
            # TODO: fill table from paper
    ],
        [  # Ceiling absorbers
            # TODO: fill table from paper
    ],
        [  # Special absorbers
            # TODO: fill table from paper
    ],
    ]

    @staticmethod
    def getAbsorptionCoefficients(category, material, units='dB'):
        category = category.lower().strip()
        if category not in MaterialAbsorptionTable.categories:
            raise Exception(
                'Unknown category for material absorption table: %s' % (category))
        categoryIdx = MaterialAbsorptionTable.categories.index(category)

        material = material.lower().strip()
        if material not in MaterialAbsorptionTable.materials[categoryIdx]:
            raise Exception('Unknown material for category %s in material absorption table: %s' % (
                category, material))
        materialIdx = MaterialAbsorptionTable.materials[categoryIdx].index(
            material)

        coefficients = np.array(
            MaterialAbsorptionTable.table[categoryIdx][materialIdx])
        frequencies = np.array(AirAttenuationTable.frequencies)

        if units == 'dB':
            eps = np.finfo(np.float).eps
            coefficients = 20.0 * np.log10(1.0 - coefficients + eps)
        elif units == 'normalized':
            # Nothing to do
            pass
        else:
            raise Exception('Unsupported units: %s' % (units))
        return coefficients, frequencies
class AirAttenuationTable(object):
    # From: Auralization : fundamentals of acoustics, modelling, simulation,
    # algorithms and acoustic virtual reality

    temperatures = [10.0, 20.0]
    relativeHumidities = [40.0, 60.0, 80.0]
    frequencies = [31.5, 63, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]

    # # Air attenuation coefficient, in 10^-3 / m
    # table = [[  # 10 deg C
    #     [0.1, 0.2, 0.5, 1.1, 2.7, 9.4, 29.0],  # 30-50% hum
    #     [0.1, 0.2, 0.5, 0.8, 1.8, 5.9, 21.1],  # 50-70% hum
    #     [0.1, 0.2, 0.5, 0.7, 1.4, 4.4, 15.8],  # 70-90% hum
    # ],
    #     [  # 20 deg C
    #         [0.1, 0.3, 0.6, 1.0, 1.9, 5.8, 20.3],  # 30-50% hum
    #         [0.1, 0.3, 0.6, 1.0, 1.7, 4.1, 13.5],  # 50-70% hum
    #         [0.1, 0.3, 0.6, 1.1, 1.7, 3.5, 10.6],  # 70-90% hum
    # ]
    # ]

    # @staticmethod
    # def getAttenuations(distance, temperature, relativeHumidity, units='dB'):
    #     closestTemperatureIdx = np.argmin(
    #         np.sqrt((np.array(AirAttenuationTable.temperatures) - temperature) ** 2))
    #     closestHumidityIdx = np.argmin(
    #         np.sqrt((np.array(AirAttenuationTable.relativeHumidities) - relativeHumidity) ** 2))
    #
    #     attenuations = np.array(
    #         AirAttenuationTable.table[closestTemperatureIdx][closestHumidityIdx])
    #     frequencies = np.array(AirAttenuationTable.frequencies)
    #     eps = np.finfo(np.float).eps
    #     attenuations = np.clip(distance * 1e-3 * attenuations, 0.0, 1.0 - eps)
    #     if units == 'dB':
    #         eps = np.finfo(np.float).eps
    #         attenuations = 20.0 * np.log10(1.0 - attenuations + eps)
    #     elif units == 'normalized':
    #         # Nothing to do
    #         pass
    #     else:
    #         raise Exception('Unsupported units: %s' % (units))
    #     return attenuations, frequencies
    @staticmethod
    def getAttenuations(distance, temperature, relativeHumidity, units='dB'):

        frequencies = np.array(AirAttenuationTable.frequencies)
        ratios = np.array(absorptCoeff(frequencies,temperature,relativeHumidity,P_atm= 101.325, unit="ratio"))

        eps = np.finfo(np.float).eps
        attenuations = np.clip(distance * 1e-3 * ratios, 0.0, 1.0 - eps)
        # any values smaller than 0.0 becomes 0.0, larger than 1.0-eps become larger than 1.0-eps

        if units == 'dB':
            eps = np.finfo(np.float).eps
            attenuations = 20.0 * np.log10(1.0 - attenuations + eps)
        elif units == 'normalized':
            # Nothing to do
            pass
        else:
            raise Exception('Unsupported units: %s' % (units))

        return attenuations, frequencies



# This is created for the CR2 room with expanded frequency range to match up the CR2 collected setting
# class MaterialAbsorptionTable(object):
#     # From: Auralization : fundamentals of acoustics, modelling, simulation, algorithms and acoustic virtual reality
#     # https://cds.cern.ch/record/1251519/files/978-3-540-48830-9_BookBackMatter.pdf
#
#     categories = ['hard surfaces', 'glazing','ceiling', 'floor','coverings']
#     # frequencies = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
#     frequencies = [20,     25.0,     31.5,    40.0,    50.0,    63.0,   80.0,   100.0,
#                    125.0,  160.0,    200.0,   250.0,   315.0,   400.0,  500.0,  630.0,
#                    800.0,  1000.0,   1250.0,  1600.0,  2000.0,  2500.0, 3150.0, 4000.0,
#                    5000.0, 6300.0,   8000.0,  10000.0, 12500.0, 16000,  20000]
#     materials = [[  # hard surfaces, such as concrete,
#         "mat_cr2_concrete",
#     ],
#         [  # Glazing
#             "mat_cr2_windows",  # Glass window
#     ],
#         [  # ceiling
#             "mat_cr2_ceiling",  # Ceiling
#     ],
#         [  # Floor
#             "mat_cr2_floor",
#     ],
#         [  # coverings, such as plaster and curtains
#             "mat_cr2_plaster"
#     ],
#     ]
#
#     # Tables of random-incidence absorption coefficients
#     table = [[  # hard surface
#         [0.049, 0.051, 0.053, 0.065, 0.075, 0.077, 0.105, 0.099,
#          0.085, 0.091, 0.081, 0.075, 0.05,  0.05,  0.056, 0.057,
#          0.06, 0.059,  0.055, 0.056, 0.059, 0.061, 0.051, 0.044,
#          0.037, 0.035, 0.041, 0.045, 0.05, 0.037,  0.01], # mat_CR2_concrete
#
#     ],
#         [  # glazing
#             [0.317, 0.266, 0.263, 0.243, 0.247, 0.223, 0.294, 0.221,
#              0.175, 0.167, 0.134, 0.073, 0.052, 0.054, 0.049, 0.034,
#              0.048, 0.057, 0.076, 0.076, 0.133, 0.106, 0.058, 0.055,
#              0.061, 0.058, 0.053, 0.049, 0.054, 0.037, 0.01], # mat_CR2_windows
#     ],
#         [  # ceiling
#             [0.108, 0.091, 0.068, 0.04,  0.021, 0.007, 0.032, 0.063,
#              0.083, 0.111, 0.107, 0.104, 0.07,  0.061, 0.048, 0.048,
#              0.05,  0.049, 0.045, 0.045, 0.047, 0.049, 0.074, 0.062,
#              0.056, 0.053, 0.05,  0.049, 0.06,  0.049, 0.016], # mat_CR2_ceiling
#     ],
#         [  # Floor
#             [0.041, 0.056, 0.07,  0.082, 0.089, 0.086, 0.096, 0.084,
#              0.071, 0.082, 0.082, 0.091, 0.071, 0.072, 0.07,  0.067,
#              0.069, 0.065, 0.062, 0.062, 0.062, 0.063, 0.051, 0.043,
#              0.036, 0.035, 0.033, 0.032, 0.037, 0.029, 0.009],
#     ],
#         [  # coverings
#             [0.045, 0.059, 0.073, 0.082, 0.083, 0.075, 0.073, 0.051,
#              0.033, 0.034, 0.038, 0.05,  0.043, 0.04,  0.039, 0.043,
#              0.046, 0.044, 0.043, 0.044, 0.048, 0.051, 0.043, 0.036,
#              0.031, 0.031, 0.028, 0.045, 0.05,  0.036, 0.01], # mat_CR2_plaster
#     ],
#         [  # Wall absorbers
#             # TODO: fill table from paper
#     ],
#         [  # Ceiling absorbers
#             # TODO: fill table from paper
#     ],
#         [  # Special absorbers
#             # TODO: fill table from paper
#     ],
#     ]
#
#     @staticmethod
#     def getAbsorptionCoefficients(category, material, units='dB'):
#         category = category.lower().strip()
#         if category not in MaterialAbsorptionTable.categories:
#             raise Exception(
#                 'Unknown category for material absorption table: %s' % (category))
#         categoryIdx = MaterialAbsorptionTable.categories.index(category)
#
#         material = material.lower().strip()
#         if material not in MaterialAbsorptionTable.materials[categoryIdx]:
#             raise Exception('Unknown material for category %s in material absorption table: %s' % (
#                 category, material))
#         materialIdx = MaterialAbsorptionTable.materials[categoryIdx].index(
#             material)
#
#         coefficients = np.array(
#             MaterialAbsorptionTable.table[categoryIdx][materialIdx])
#         frequencies = np.array(AirAttenuationTable.frequencies)
#
#         if units == 'dB':
#             eps = np.finfo(np.float).eps
#             coefficients = 20.0 * np.log10(1.0 - coefficients + eps)
#         elif units == 'normalized':
#             # Nothing to do
#             pass
#         else:
#             raise Exception('Unsupported units: %s' % (units))
#         return coefficients, frequencies
# class AirAttenuationTable(object):
#     # From: Auralization : fundamentals of acoustics, modelling, simulation,
#     # algorithms and acoustic virtual reality
#     # Note: THe expansion is done merely for evaluation on the simulation with real collected data
#     ##   which was collected under 19.5C temperature and 41.7% relative humidity. Further work can be done
#     ##   in the future.
#
#     ### The coefficients "alpha" from The calcualtor is in dB unit that is absorpted, which needs to be converted according to:
#     # alpha = 20 log(P(absorpted)/P0)
#     temperatures = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
#     relativeHumidities = [40.0, 42.0, 44.0, 46.0, 48.0, 50.0]
#     frequencies = [20,     25.0,     31.5,    40.0,    50.0,    63.0,   80.0,   100.0,
#                    125.0,  160.0,    200.0,   250.0,   315.0,   400.0,  500.0,  630.0,
#                    800.0,  1000.0,   1250.0,  1600.0,  2000.0,  2500.0, 3150.0, 4000.0,
#                    5000.0, 6300.0,   8000.0,  10000.0, 12500.0, 16000,  20000]
#
#     # Air attenuation coefficient, in 10^-3 / m
#
#     @staticmethod
#     def getAttenuations(distance, temperature, relativeHumidity, units='dB'):
#
#         frequencies = np.array(AirAttenuationTable.frequencies)
#         ratios = np.array(absorptCoeff(frequencies,temperature,relativeHumidity,P_atm= 101.325, unit="ratio"))
#
#         eps = np.finfo(np.float).eps
#         attenuations = np.clip(distance * 1e-3 * ratios, 0.0, 1.0 - eps)
#         # any values smaller than 0.0 becomes 0.0, larger than 1.0-eps become larger than 1.0-eps
#
#         if units == 'dB':
#             eps = np.finfo(np.float).eps
#             attenuations = 20.0 * np.log10(1.0 - attenuations + eps)
#         elif units == 'normalized':
#             # Nothing to do
#             pass
#         else:
#             raise Exception('Unsupported units: %s' % (units))
#
#         return attenuations, frequencies



def getAcousticModelNodeForModel(model, mode='box'):
    transform = TransformState.makeIdentity()
    if mode == 'mesh':
        acousticModel = model.copyTo(model.getParent())
        acousticModel.detachNode()

    elif mode == 'box':
        # Bounding box approximation
        minRefBounds, maxRefBounds = model.getTightBounds()
        refDims = maxRefBounds - minRefBounds
        refPos = model.getPos()
        refCenter = minRefBounds + (maxRefBounds - minRefBounds) / 2.0
        refDeltaCenter = refCenter - refPos

        acousticModel = loadModel(os.path.join(MODEL_DATA_DIR, 'cube.egg'))

        # Rescale the cube model to match the bounding box of the original
        # model
        minBounds, maxBounds = acousticModel.getTightBounds()
        dims = maxBounds - minBounds
        pos = acousticModel.getPos()
        center = minBounds + (maxBounds - minBounds) / 2.0
        deltaCenter = center - pos

        position = refPos + refDeltaCenter - deltaCenter
        scale = LVector3f(refDims.x / dims.x, refDims.y /
                          dims.y, refDims.z / dims.z)
        transform = TransformState.makePos(position).compose(
            TransformState.makeScale(scale))

        # TODO: validate applied transform here

    else:
        raise Exception(
            'Unknown mode type for acoustic object shape: %s' % (mode))

    acousticModel.setName(model.getName())
    acousticModel.setTransform(acousticModel.getTransform().compose(transform))

    return acousticModel


class AcousticImpulseResponse(object):
    def __init__(self, impulse, samplingRate, source, target):
        self.__dict__.update(impulse=impulse, samplingRate=samplingRate,
                             source=source, target=target)

        # EVERT instance
        self.solution = None

class EvertAudioSound(object):
    # Python implementation of AudioSound abstract class:
    # https://www.panda3d.org/reference/1.9.4/python/panda3d.core.AudioSound

    def __init__(self, filename):
        self.name = os.path.basename(filename)
        self.filename = filename

        # Load sound from file
        fs,data = wavfile.read(filename)

        # Make sure the sound is mono, and keep the first channel only
        if data.ndim == 2 and np.maximum(data.shape[0], data.shape[0]) >= 1:
            if data.shape[0]>data.shape[1]:
                data=np.transpose(data)
                data = np.sum(data,axis=0)
            else:
                data = np.sum(data,axis=0)

        # Normalize in the interval [-1, 1]
        data = data / np.max(np.abs(data))
        self.data = data
        self.samplingRate = fs

        self.reset()

    def reset(self):

        self.t = 0.0
        self.isActive = True
        self.isLoop = False
        self.loopCount = 1
        self.curLoopCount = 0
        self.priority = 0
        self.volume = 1.0
        self.playRate = 1.0
        self.balance = 0.0
        self.curStatus = AudioSound.READY

    def resample(self, newSamplingRate):
        if self.samplingRate != newSamplingRate:
            N = self.data.shape[-1]
            nbSamples = int(N * newSamplingRate / self.samplingRate)
            self.data = scipy.signal.resample(self.data, nbSamples, axis=-1)
            self.samplingRate = newSamplingRate

    def configureFilters(self, config):
        raise NotImplementedError()

    def get3dMaxDistance(self):
        raise NotImplementedError()

    def get3dMinDistance(self):
        raise NotImplementedError()

    def getActive(self):
        return self.isActive

    def getBalance(self):
        return self.balance

    def getFinishedEvent(self):
        raise NotImplementedError()

    def getLoop(self):
        return self.isLoop

    def getLoopCount(self):
        return self.loopCount

    def getName(self):
        return self.name

    def getPlayRate(self):
        return self.playRate

    def getPriority(self):
        return self.priority

    def getSpeakerLevel(self, index):
        raise NotImplementedError()

    def getSpeakerMix(self, speaker):
        raise NotImplementedError()

    def getTime(self):
        return self.t / self.length()

    def getVolume(self):
        return self.volume

    def length(self):
        return len(self.data) / float(self.samplingRate)

    def output(self, out):
        raise NotImplementedError()

    def play(self):
        self.curStatus = AudioSound.PLAYING


    def set3dAttributes(self, px, py, pz, vx, vy, vz):
        raise NotImplementedError()

    def set3dMaxDistance(self, dist):
        raise NotImplementedError()

    def set3dMinDistance(self, dist):
        raise NotImplementedError()

    def setActive(self, flag):
        self.isActive = flag

    def setBalance(self, balance_right):
        raise NotImplementedError()

    def setFinishedEvent(self, event):
        raise NotImplementedError()

    def setLoop(self, loop):
        self.isLoop = loop

    def setLoopCount(self, loop_count):
        self.loopCount = loop_count

    def setPlayRate(self, play_rate):
        raise NotImplementedError()

    def setPriority(self, priority):
        self.priority = priority

    def setSpeakerLevels(self, level1, level2, level3, level4, level5,
                         level6, level7, level8, level9):
        raise NotImplementedError()

    def setSpeakerMix(self, frontleft, frontright, center, sub, backleft, backright, sideleft, sideright):
        raise NotImplementedError()

    def setTime(self, start_time):
        assert start_time >= 0.0 and start_time <= 1.0
        self.t = start_time * self.length()

    def setVolume(self, volume):
        self.volume = volume

    def status(self):
        return self.curStatus

    def stop(self):
        # Reset seek position to beginning
        self.t = 0.0
        self.curStatus = AudioSound.READY

    def write(self, out):
        raise NotImplementedError()


# XXX: use Task chains for multithreading, see
# https://www.panda3d.org/manual/index.php/Task_Chains
class AudioPlayer(object):

    def __init__(self, acoustics):

        self.acoustics = acoustics

        import pyaudio
        from pyaudio import paFloat32

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=paFloat32,
                                  channels=acoustics.getNbOutputChannels(),
                                  rate=int(acoustics.samplingRate),
                                  output=True)

        # Select default agent
        self.agentName = self.acoustics.scene.agents[0].getName()

        taskMgr.add(self.update, 'audio-player', sort=1)

        # Get a pointer to Panda's global ClockObject, used for
        # synchronizing events between Python and C.
        globalClock = ClockObject.getGlobalClock()
        globalClock.setMode(ClockObject.MNonRealTime)
        globalClock.setDt(1)

        # Now we can make the TaskManager start using the new globalClock.
        taskMgr.globalClock = globalClock

    def _getFrame(self):

        obs = self.acoustics.getObservationsForAgent(self.agentName)

        if 'audio-buffer-left' in obs and 'audio-buffer-right' in obs:
            data = np.array([obs['audio-buffer-left'],
                             obs['audio-buffer-right']], dtype=np.float32)
        elif 'audio-buffer-0' in obs:
            data = np.array(obs['audio-buffer-0'], dtype=np.float32)
        else:
            raise Exception('Unknown channel configuration in observations')

        # Interleave channels if multiple
        frame = np.empty((data.shape[-1] * data.shape[0],), dtype=data.dtype)
        for i in range(data.shape[0]):
            frame[i::data.shape[0]] = data[i, :]

        return frame

    def setListeningToAgent(self, name):
        self.agentName = name

    def update(self, task):

        data = self._getFrame()
        self.stream.write(data)

        return task.cont

    def __del__(self):
        taskMgr.remove('audio-player')

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class EvertAcoustics_jz(World):
    # NOTE: the model ids of objects that correspond to opened doors. They
    # will be ignored in the acoustic scene.
    openedDoorModelIds = [
        # Doors
        '122', '133', '214', '246', '247', '361', '73', '756', '757', '758', '759', '760',
        '761', '762', '763', '764', '765', '768', '769', '770', '771', '778', '779', '780',
        's__1762', 's__1763', 's__1764', 's__1765', 's__1766', 's__1767', 's__1768', 's__1769',
        's__1770', 's__1771', 's__1772', 's__1773',
        # Curtains
        '275'
    ]

    rayColors = [
        (1.0, 1.0, 0.0, 0.2),  # yellow
        (0.0, 0.0, 1.0, 0.2),  # blue
        (0.0, 1.0, 0.0, 0.2),  # green
        (0.0, 1.0, 1.0, 0.2),  # cyan
        (1.0, 0.0, 1.0, 0.2),  # magenta
        (1.0, 0.0, 0.0, 0.2),  # red
    ]

    minRayRadius = 0.01  # m
    maxRayRadius = 0.1  # m

    def __init__(self, scene, hrtf=None, samplingRate=44100, maximumOrder=2, ray_tracing=True,
                 microphoneTransform=None,  minWidthThresholdPolygons=0.0, maxImpulseLength=1.0,
                 maxBufferLength=1.5,
                 cameraMask=BitMask32.allOn(),delay = 0):

        super(EvertAcoustics_jz, self).__init__()

        self.__dict__.update(scene=scene, hrtf=hrtf, samplingRate=samplingRate, maximumOrder=maximumOrder,ISM = True, ray_tracing=True,
                             microphoneTransform=microphoneTransform,
                             minWidthThresholdPolygons=minWidthThresholdPolygons, maxImpulseLength=maxImpulseLength,
                             maxBufferLength=maxBufferLength,cameraMask=cameraMask, delay=delay)

        self.samplingRate = samplingRate
        self.maximumOrder = maximumOrder ### maximum order of ISM model
        self.ray_tracing = ray_tracing
        self.delay = delay
        self.solutions = dict()

        self.sounds = dict()
        self.srcBuffers = dict()
        self.outBuffers = dict()
        self.ambientSounds = [] ## not quite useful for now


        if self.hrtf is not None:
            self.hrtf.resample(samplingRate)
        self.nbMicrophones = len(microphoneTransform)

        self.setAirConditions()
        self.coefficientsForMaterialId = []

        self.agents = []
        self.sources = dict()
        self.acousticImpulseResponses = []

        self.initPraDone = False

        self.world = self._initLayoutModels()
        # self._initObjects() # not used yet as all models are considered as room layout now
        self._initAgents()

        # self._initPathSolutions()
        self.scene.worlds['acoustics'] = self

        self.lastTaskTime = 0.0
        # taskMgr.add(self.update, 'acoustics', sort=0)

    def showRoomLayout(self, showCeilings=True, showWalls=True, showFloors=True):

        for np in self.scene.scene.findAllMatches('**/layouts/*ceiling'):
            if showCeilings:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

        for np in self.scene.scene.findAllMatches('**/layouts/*wall*'):
            if showWalls:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

        for np in self.scene.scene.findAllMatches('***/layouts/*floor'):
            if showFloors:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

    def destroy(self):
        # Nothing to do
        pass

    def getNbOutputChannels(self):
        if self.hrtf is not None:
            nbOutputChannels = len(self.hrtf.channels)
        else:
            nbOutputChannels = 1
        return nbOutputChannels

    def getNbMicrophones(self):
        return self.nbMicrophones


    def _loadSphereModel(self, refCenter, radius, color=(1.0, 0.0, 0.0, 1.0)):

        model = loadModel(os.path.join(MODEL_DATA_DIR, 'sphere.egg'))

        # Rescale the cube model to match the bounding box of the original
        # model
        minBounds, maxBounds = model.getTightBounds()
        dims = maxBounds - minBounds
        pos = model.getPos()
        center = minBounds + (maxBounds - minBounds) / 2.0
        deltaCenter = center - pos

        position = refCenter - deltaCenter
        model.setPos(position)

        scale = LVector3f(radius / dims.x, radius / dims.y, radius / dims.z)
        model.setScale(scale)

        # Validate approximation
        eps = 1e-4
        minBounds, maxBounds = model.getTightBounds()
        center = minBounds + (maxBounds - minBounds) / 2.0
        dims = maxBounds - minBounds
        assert np.allclose([center.x, center.y, center.z],
                           [refCenter.x, refCenter.y, refCenter.z],
                           atol=eps)

        material = Material()
        material.setAmbient(color)
        material.setDiffuse(color)
        model.clearMaterial()
        model.setMaterial(material, 1)

        return model

    def setAirConditions(self, pressureAtm=1.0, temperature=19.5, relativeHumidity=41.7):
        self.pressureAtm = pressureAtm
        self.temperature = temperature
        self.relativeHumidity = relativeHumidity


    ### calculate impuse response for evert agents
    def calculateImpulseResponse(self, srcName, agentName):

        impulse = dict()
        micNps = dict()

        if self.world.simulator_state["ism_needed"] and not self.world.simulator_state["ism_done"]:
            self.world.image_source_model()

        if self.world.simulator_state["rt_needed"] and not self.world.simulator_state["rt_done"]:
            self.world.ray_tracing()

        agentNp = self.scene.scene.find('**/' + agentName)
        sourceNp = self.scene.scene.find('**/' + srcName)


        for i in range(self.nbMicrophones):

            micName = agentName + '-mic'+str(i)

            impulse[i] = self._calculateImpulseResponse(srcName, micName)
            micNps[i] = agentNp.find('**/acoustics/microphoneNp-'+str(i))


        return AcousticImpulseResponse(impulse, self.samplingRate, sourceNp, micNps)

    ### calculate Impulse for every microphones (if has hrtf, add in multiple channels )
    def _calculateImpulseResponse(self, srcName, micName):

        # Calculate impulse response in time domain, accounting for frequency-dependent absorption,
        # then when it is time to convolve with the HRTF, do it in fourrier
        # domain.
        srcId = int(srcName.split("-")[-1])
        micId = int(micName.split("-")[1])+int(micName.split("c")[-1])

        volume_room = self.world.get_volume()
        src = self.world.sources[srcId]
        mic = self.world.mic_array.R.T[micId]
        imp = self.world.calculate_rir(micId,mic,srcId,src,volume_room)
        ### This part should be for HRTF:
        # for i in range(self.getNbOutputChannels()):
        #     ####
        #     ####

        ### trim the length of the impulse response
        # N = int(self.maxImpulseLength*self.samplingRate)
        # if imp.shape[0] > N:
        #     imp = imp[:N]  ### assume no hrtf now
        return imp

    def _initLayoutModels(self):
        LayoutModels = []
        # Load layout objects as meshes
        for model in self.scene.scene.findAllMatches('**/layouts/object*/model*'):

            model.getParent().setTag('acoustics-mode', 'obstacle')
            coefficients = TextureAbsorptionTable.getMeanAbsorptionCoefficientsFromModel(
                model, units='normalized')
            materialId = len(self.coefficientsForMaterialId)
            self.coefficientsForMaterialId.append(coefficients)

            acousticModel = getAcousticModelNodeForModel(model, mode='mesh')
            acousticModel.hide(BitMask32.allOn())
            acousticModel.show(self.cameraMask)

            objectNp = model.getParent()
            acousticsNp = objectNp.attachNewNode('acoustics')
            acousticModel.reparentTo(acousticsNp)

            material = Material()
            intensity = np.mean(1.0 - coefficients)
            material.setAmbient((intensity, intensity, intensity, 1))
            material.setDiffuse((intensity, intensity, intensity, 1))
            acousticModel.clearMaterial()
            acousticModel.setMaterial(material, 1)
            acousticModel.setTextureOff(1)
            acousticModel.setTag('materialId', str(materialId))

            parent = objectNp.find('**/physics*')
            if parent.isEmpty():
                parent = objectNp
            acousticsNp.reparentTo(parent)

            Polys = self._addModelGeometry(acousticModel)
            for Poly in Polys:
                LayoutModels.append(Poly)
        return pra.Room(LayoutModels,max_order= self.maximumOrder,temperature = self.temperature, humidity= self.relativeHumidity,
                        ray_tracing=self.ray_tracing,t0=self.delay,fs = self.samplingRate)

    def _initAgents(self):

        # Load agents
        for agent in self.scene.scene.findAllMatches('**/agents/agent*'):

            agent.setTag('acoustics-mode', 'listener')

            acousticNode = agent.attachNewNode('acoustics')
            self.outBuffers[agent.getName()] = dict()
            if self.microphoneTransform is not None:
                for i in range(self.getNbMicrophones()):
                    MicNodeName = 'microphoneNp-%s' % (i)

                    # Load model for the microphone
                    MicNodeNp = acousticNode.attachNewNode(MicNodeName)
                    MicNodeNp.setTransform(self.microphoneTransform[i])
                    model = self._loadSphereModel(refCenter=LVecBase3(
                        0.0, 0.0, 0.0), radius=0.10, color=(1.0, 0.0, 0.0, 1.0))
                    model.reparentTo(MicNodeNp)

                    # Add listeners for pyroomacoustics
                    netMat = MicNodeNp.getNetTransform().getMat()
                    lstNetPos = netMat.getRow3(3)
                    lstNetMat = netMat.getUpper3()
                    loc = np.array((lstNetPos.x ,lstNetPos.y , lstNetPos.z))

                    self.world.add_microphone(loc)
                    # # Allocate output buffer
                    self.outBuffers[agent.getName()][i] = np.array(
                        [[] for _ in range(self.getNbOutputChannels())])
                    # logger.debug('Agent %s: microphone at position (x=%f, y=%f, z=%f)' % (
                    #     agent.getName(), lstNetPos.x, lstNetPos.y, lstNetPos.z))

            self.agents.append(agent)


    def _initObjects(self):

        # Load objects
        for model in self.scene.scene.findAllMatches('**/objects/object*/model*'):
            modelId = model.getParent().getTag('model-id')
            if modelId in self.openedDoorModelIds:
                continue

            # Check if object is static
            isStatic = True
            if model.hasNetTag('physics-mode'):
                # Ignore dynamic models
                if model.getNetTag('physics-mode') == 'dynamic':
                    isStatic = False

            isObstacle = True
            if model.hasNetTag('acoustics-mode'):
                if model.getNetTag('acoustics-mode') != 'obstacle':
                    isObstacle = False

            if isObstacle and isStatic:

                model.getParent().setTag('acoustics-mode', 'obstacle')

                coefficients = TextureAbsorptionTable.getMeanAbsorptionCoefficientsFromModel(
                    model, units='normalized')
                materialId = len(self.coefficientsForMaterialId)
                self.coefficientsForMaterialId.append(coefficients)

                acousticModel = getAcousticModelNodeForModel(
                    model, mode=self.objectMode)
                acousticModel.hide(BitMask32.allOn())
                acousticModel.show(self.cameraMask)

                objectNp = model.getParent()
                acousticsNp = objectNp.attachNewNode('acoustics')
                acousticModel.reparentTo(acousticsNp)

                material = Material()
                intensity = np.mean(1.0 - coefficients)
                material.setAmbient((intensity, intensity, intensity, 1))
                material.setDiffuse((intensity, intensity, intensity, 1))
                acousticModel.clearMaterial()
                acousticModel.setMaterial(material, 1)
                acousticModel.setTextureOff(1)
                acousticModel.setTag('materialId', str(materialId))

                # NOTE: since the object is static, we don't even need to
                # reparent the acoustic node under physics
                parent = objectNp.find('**/physics*')
                if parent.isEmpty():
                    parent = objectNp
                acousticsNp.reparentTo(parent)

                self._addModelGeometry(acousticModel)

    def _addModelGeometry(self, model):

        if model.hasNetTag('physics-mode') and model.getNetTag('physics-mode') != 'static':
            raise Exception('Obstacles in EVERT must be static!')
        ## need to take the material now
        materialId = int(model.getTag('materialId'))

        # Add polygons to pyroomacoustics engine
        # TODO: for bounding box approximations, we could reduce the number of triangles by
        # half if each face of the box was modelled as a single rectangular
        # polygon.
        Polys, triangles = getAcousticPolysFromModel(model)
        for Poly, triangle in zip(Polys, triangles):

            # Validate triangle (ignore if area is too small)
            dims = np.max(triangle, axis=0) - np.min(triangle, axis=0)
            s = np.sum(
                np.array(dims > self.minWidthThresholdPolygons, dtype=np.int))
            if s < 2:
                continue
        return Polys

    def _updateListeners(self):

        # Update positions of listeners
        for index, agent in enumerate(self.agents):
            for i in range (self.nbMicrophones):
                MicNodeName='microphoneNp-%s' % (i)
                microphoneNp = agent.find('**/%s'% (MicNodeName))

                # now set up the location of the mic according to the absolute location of the microphoneNp
                netMat = microphoneNp.getNetTransform().getMat()
                lstNetPos = netMat.getRow3(3)
                lstNetMat = netMat.getUpper3() ## this is for the direction, which is not used now.
                self.world.mov_microphone(index+i, [lstNetPos.x , lstNetPos.y, lstNetPos.z])  # m to mm

                # # NOTE Orientation matrix: first column is direction vector, second
                # # column is up vector, third column is right vector
                # lst.setOrientation(Matrix3(lstNetMat.getCell(0, 0), lstNetMat.getCell(2, 0), -lstNetMat.getCell(1, 0),
                #                        lstNetMat.getCell(0, 1), lstNetMat.getCell(
                #                            2, 1), -lstNetMat.getCell(1, 1),
                #                        lstNetMat.getCell(0, 2), lstNetMat.getCell(2, 2), -lstNetMat.getCell(1, 2)))
                #
                # # logger.debug('Agent %s: microphone at position (x=%f, y=%f, z=%f)' % (
                # #     agent.getName(), lstNetPos.x, lstNetPos.y, lstNetPos.z))

    def _updateSources(self):

        # NOTE: no need to update position of source, since they are static for now
        for objName, _ in iteritems(self.sounds):

            obj = self.scene.scene.find('**/objects/%s' % (objName))

            src = None
            for i in range(self.world.numSources()):
                source = self.world.getSource(i)
                if source.getName() == str(obj.getName()):
                    src = source
                    break
            assert src is not None

            # Get position and orientation of the static sound source
            sourceNetTrans = obj.getNetTransform().getMat()
            sourceNetPos = sourceNetTrans.getRow3(3)

            logger.debug('Static source %s: at position (x=%f, y=%f, z=%f)' % (
                obj.getName(), sourceNetPos.x, sourceNetPos.y, sourceNetPos.z))

    def step(self, dt):

        # self._updateSources() # switch off now as no change of the source to reduce the computation
        self._updateListeners()
        self.world.ray_tracing()
        self.world.image_source_model()
        self._updateSrcBuffers(dt)
        # self._updateOutputBuffers(dt)



    def getObservationsForAgent(self, name, clearBuffer=True):

        observations = dict()
        for i in range(self.nbMicrophones):

            if self.hrtf is not None:
                channelNames = self.hrtf.channels
            else:
                channelNames = ['0']

            micName = name+'-mic'+str(i)
            observations[micName]=dict()

            for j, channelName in enumerate(channelNames):
                observations[micName]['audio-buffer-%s' %
                            (channelName)] = self.outBuffers[name][i][j]

            if clearBuffer:
                # Clear output buffer
                self.outBuffers[name][i] = np.array(
                    [[] for _ in range(self.getNbOutputChannels())])

        return observations

    def addAmbientSound(self, sound):
        if sound not in self.ambientSounds:
            self.ambientSounds.append(sound)

            # Allocate a buffer for that sound
            nbMaxSamples = int(self.maxBufferLength * self.samplingRate)
            self.srcBuffers[sound] = np.zeros((nbMaxSamples,), np.float32)

    def attachSoundToObject(self, sound, obj):
        """
        Sound will come from the location of the object it is attached to
        """

        if obj.hasTag('physics-mode') and obj.getTag('physics-mode') != 'static':
            raise Exception('Sources in EVERT must be static!')
        sound.resample(self.samplingRate)
        self.sounds[obj.getName().split("-")[1]+"-"+obj.getName().split("-")[2]] = sound
        # Allocate a buffer for that sound
        nbMaxSamples = int(self.maxBufferLength * self.samplingRate)
        self.srcBuffers[sound] = np.zeros((nbMaxSamples,), np.float32)


        # Get position and orientation of the static sound source
        sourceNetTrans = obj.getNetTransform().getMat()
        sourceNetPos = sourceNetTrans.getRow3(3)
        sourceNetMat = sourceNetTrans.getUpper3()
        loc= np.array((sourceNetPos.x,sourceNetPos.y,sourceNetPos.z))


        self.world.add_soundsource(sound)

        # NOTE Orientation matrix: first column is direction vector, second
        # column is up vector, third column is right vector
        # src.setOrientation(Matrix3(sourceNetMat.getCell(0, 0), sourceNetMat.getCell(2, 0), -sourceNetMat.getCell(1, 0),
        #                            sourceNetMat.getCell(0, 1), sourceNetMat.getCell(
        #                                2, 1), -sourceNetMat.getCell(1, 1),
        #                            sourceNetMat.getCell(0, 2), sourceNetMat.getCell(2, 2), -sourceNetMat.getCell(1, 2)))
        #
        # logger.debug('Static source %s: at position (x=%f, y=%f, z=%f)' % (
        #     obj.getName(), sourceNetPos.x, sourceNetPos.y, sourceNetPos.z))

        # Add acoustic node in scene graph
        obj.setTag('acoustics-mode', 'source')
        acousticsNp = obj.attachNewNode('acoustics')

        return 1


    def _updateSrcBuffers(self, dt):

        if dt > 0.0:

            # NOTE: round dt to make sure it equals an non-fractional number of
            # samples
            dt = float(int(np.floor(dt * self.samplingRate)) /
                       self.samplingRate)
            newDataLength = int(np.round(dt * self.samplingRate))

            # Gather all attached and ambient sounds
            sounds = []
            for sound in itervalues(self.sounds):
                sounds.append(sound)
            sounds.extend(self.ambientSounds)

            for sound in sounds:

                # TODO: handle isActive() property of sounds

                # Check sound status for updating audio buffer
                if sound.status() == AudioSound.PLAYING:

                    # Check the current seek position for the sound and get
                    # data
                    tStart = sound.t
                    tEnd = sound.t + dt

                    if tEnd <= sound.length():
                        # Get the sound data for this timestep
                        startIdx = int(np.round(tStart * sound.samplingRate))
                        endIdx = int(np.round(tEnd * sound.samplingRate)) - 1
                        data = sound.signal[startIdx:endIdx + 1]
                    else:
                        # Get the sound data for this timestep
                        startIdx = int(tStart * sound.samplingRate)
                        data = sound.signal[startIdx:]

                        # Pad end with zeros to match desired buffer length
                        data = np.pad(
                            data, [(0, newDataLength - len(data))], mode='constant')

                        # Indicates end of play loop
                        sound.curLoopCount += 1
                        sound.stop()

                    # Update buffer for sound
                    assert len(data) == newDataLength
                    buf = self.srcBuffers[sound]
                    if newDataLength < len(buf):
                        # Shift buffer to make place for new data
                        buf[:len(buf) - len(data)] = buf[len(data):]
                        buf[len(buf) - len(data):] = data
                    else:
                        buf = data[len(data) - len(buf):]
                    self.srcBuffers[sound] = buf

                    # Update seek position for sound
                    sound.t += dt

                elif sound.status() == AudioSound.READY:

                    # Update buffer for sound
                    buf = self.srcBuffers[sound]
                    data = np.zeros((newDataLength,), dtype=buf.dtype)
                    if newDataLength < len(buf):
                        # Shift buffer to make place for new data
                        buf[:len(buf) - len(data)] = buf[len(data):]
                        buf[len(buf) - len(data):] = data
                    else:
                        buf = data[len(data) - len(buf):]
                    self.srcBuffers[sound] = buf

                # Handling of loop mode for sounds
                if sound.status() == AudioSound.READY:
                    if sound.getLoop():
                        # Check loop count: 0 = forever; 1 = play once; n =
                        # play n times
                        if sound.getLoopCount() > 0:
                            if sound.curLoopCount < sound.getLoopCount():
                                # Not yet reach the maximum replay limit
                                sound.play()

    def _updateOutputBuffers(self, dt):

        if dt > 0.0:

            # Loop for all agent
            for agent in self.agents:
                # Initialize new output buffer for this timestep
                # NOTE: round dt to make sure it equals an non-fractional
                # number of samples
                dt = float(int(np.floor(dt * self.samplingRate)) /
                           self.samplingRate)
                newDataLength = int(np.round(dt * self.samplingRate))

                # FIXME: need to take care of buffer overrun here
                if dt > self.maxBufferLength + self.maxImpulseLength:
                    newDataLength = int(
                        (self.maxBufferLength - self.maxImpulseLength) * self.samplingRate)
                nbChannels = self.getNbOutputChannels()
                outBuf = np.zeros(
                    (nbChannels, newDataLength), dtype=np.float32)
                srcName = list(self.sounds)[0]
                sound = self.sounds[srcName]

                for j in range(self.nbMicrophones):
                    micName = str(agent.getName()+'-mic'+str(j))
                    imp= self._calculateImpulseResponse(srcName, micName)
                    if imp.shape[-1] > 0:
                        buf = self.srcBuffers[sound]
                        # Convolve source with impulse and add contribution
                                # to output buffer
                        nbInSamples = newDataLength + imp.shape[-1] - 1

                        if nbChannels > 1:
                            for channel in range(nbChannels):
                                outBuf[channel, :] += signal.fftconvolve(
                                        buf[len(buf) - nbInSamples:], imp[channel], mode='valid')
                        else:
                            outBuf[:] += signal.fftconvolve(
                                        buf[len(buf) - nbInSamples:], imp, mode='valid')
                    # Loop for all ambient sounds
                    for sound in self.ambientSounds:

                    # Get the source buffer related to the sound
                        buf = self.srcBuffers[sound]

                        # Add contribution equally to all channels
                        nbChannels = self.getNbOutputChannels()
                        nbInSamples = newDataLength
                        for channel in range(nbChannels):
                                outBuf[channel,:] += buf[len(buf) - nbInSamples:] * sound.getVolume()

                    # XXX: should we concatenate to create a continuous buffer or
                    # simply append as frames to a list?
                    self.outBuffers[agent.getName()][j] = np.concatenate(
                        (self.outBuffers[agent.getName()][j], outBuf), axis=-1)




