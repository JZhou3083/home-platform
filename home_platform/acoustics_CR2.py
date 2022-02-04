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
import numpy as np
import matplotlib.pyplot as plt

from six import itervalues, iteritems
from scipy import signal
from string import digits
from evert import Room as EvertRoom
from evert import Source, Listener, Vector3, Matrix3, Polygon, PathSolution
from evert import Viewer as EvertViewer

from panda3d.core import NodePath, LVector3f, LVecBase3, Material, TransformState, AudioSound, CS_zup_right, BitMask32, ClockObject
from direct.task.TaskManagerGlobal import taskMgr

from home_platform.core import World
from home_platform.suncg import loadModel
from home_platform.rendering import get3DTrianglesFromModel, getColorAttributesFromModel
from home_platform.utils import vec3ToNumpyArray

logger = logging.getLogger(__name__)

MODEL_DATA_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "data", "models")


def getAcousticPolygonsFromModel(model):
    polygons = []
    triangles = get3DTrianglesFromModel(model)
    for triangle in triangles:
        pts = []
        for pt in triangle:
            # NOTE: EVERT works in milimeter units
            pts.append(Vector3(pt[0] * 1000.0, pt[1] * 1000.0, pt[2] * 1000.0))
        polygons.append(Polygon(pts))
    return polygons, triangles

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
# class MaterialAbsorptionTable(object):
#     # From: Auralization : fundamentals of acoustics, modelling, simulation, algorithms and acoustic virtual reality
#     # https://cds.cern.ch/record/1251519/files/978-3-540-48830-9_BookBackMatter.pdf
#
#     categories = ['hard surfaces', 'glazing','ceiling', 'floor','coverings']
#     frequencies = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
#     # frequencies = [20,     25.0,     31.5,    40.0,    50.0,    63.0,   80.0,   100.0,
#     #                125.0,  160.0,    200.0,   250.0,   315.0,   400.0,  500.0,  630.0,
#     #                800.0,  1000.0,   1250.0,  1600.0,  2000.0,  2500.0, 3150.0, 4000.0,
#     #                5000.0, 6300.0,   8000.0,  10000.0, 12500.0, 16000,  20000]
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
#         [0.085, 0.075, 0.056,  0.059,  0.059, 0.044, 0.041], # mat_CR2_concrete
#
#     ],
#         [  # glazing
#             [0.175, 0.073, 0.049, 0.057, 0.133, 0.055, 0.053], # mat_CR2_windows
#     ],
#         [  # ceiling
#             [0.083, 0.104, 0.048, 0.049, 0.047, 0.062, 0.05], # mat_CR2_ceiling
#     ],
#         [  # Floor
#             [0.071, 0.091, 0.07, 0.065, 0.062, 0.043, 0.033],
#     ],
#         [  # coverings
#             [0.033, 0.05,  0.039, 0.044, 0.048, 0.036, 0.028], # mat_CR2_plaster
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
# This is created for the CR2 room with expanded frequency range to match up the CR2 collected setting
class MaterialAbsorptionTable(object):
    # From: Auralization : fundamentals of acoustics, modelling, simulation, algorithms and acoustic virtual reality
    # https://cds.cern.ch/record/1251519/files/978-3-540-48830-9_BookBackMatter.pdf

    categories = ['hard surfaces', 'glazing','ceiling', 'floor','coverings']
    # frequencies = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    frequencies = [20,     25.0,     31.5,    40.0,    50.0,    63.0,   80.0,   100.0,
                   125.0,  160.0,    200.0,   250.0,   315.0,   400.0,  500.0,  630.0,
                   800.0,  1000.0,   1250.0,  1600.0,  2000.0,  2500.0, 3150.0, 4000.0,
                   5000.0, 6300.0,   8000.0,  10000.0, 12500.0, 16000,  20000]
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
        [0.049, 0.051, 0.053, 0.065, 0.075, 0.077, 0.105, 0.099,
         0.085, 0.091, 0.081, 0.075, 0.05,  0.05,  0.056, 0.057,
         0.06, 0.059,  0.055, 0.056, 0.059, 0.061, 0.051, 0.044,
         0.037, 0.035, 0.041, 0.045, 0.05, 0.037,  0.01], # mat_CR2_concrete

    ],
        [  # glazing
            [0.317, 0.266, 0.263, 0.243, 0.247, 0.223, 0.294, 0.221,
             0.175, 0.167, 0.134, 0.073, 0.052, 0.054, 0.049, 0.034,
             0.048, 0.057, 0.076, 0.076, 0.133, 0.106, 0.058, 0.055,
             0.061, 0.058, 0.053, 0.049, 0.054, 0.037, 0.01], # mat_CR2_windows
    ],
        [  # ceiling
            [0.108, 0.091, 0.068, 0.04,  0.021, 0.007, 0.032, 0.063,
             0.083, 0.111, 0.107, 0.104, 0.07,  0.061, 0.048, 0.048,
             0.05,  0.049, 0.045, 0.045, 0.047, 0.049, 0.074, 0.062,
             0.056, 0.053, 0.05,  0.049, 0.06,  0.049, 0.016], # mat_CR2_ceiling
    ],
        [  # Floor
            [0.041, 0.056, 0.07,  0.082, 0.089, 0.086, 0.096, 0.084,
             0.071, 0.082, 0.082, 0.091, 0.071, 0.072, 0.07,  0.067,
             0.069, 0.065, 0.062, 0.062, 0.062, 0.063, 0.051, 0.043,
             0.036, 0.035, 0.033, 0.032, 0.037, 0.029, 0.009],
    ],
        [  # coverings
            [0.045, 0.059, 0.073, 0.082, 0.083, 0.075, 0.073, 0.051,
             0.033, 0.034, 0.038, 0.05,  0.043, 0.04,  0.039, 0.043,
             0.046, 0.044, 0.043, 0.044, 0.048, 0.051, 0.043, 0.036,
             0.031, 0.031, 0.028, 0.045, 0.05,  0.036, 0.01], # mat_CR2_plaster
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
# class AirAttenuationTable(object):
#     # From: Auralization : fundamentals of acoustics, modelling, simulation,
#     # algorithms and acoustic virtual reality
#
#     temperatures = [10.0, 20.0]
#     relativeHumidities = [40.0, 60.0, 80.0]
#     frequencies = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
#
#     # Air attenuation coefficient, in 10^-3 / m
#     table = [[  # 10 deg C
#         [0.1, 0.2, 0.5, 1.1, 2.7, 9.4, 29.0],  # 30-50% hum
#         [0.1, 0.2, 0.5, 0.8, 1.8, 5.9, 21.1],  # 50-70% hum
#         [0.1, 0.2, 0.5, 0.7, 1.4, 4.4, 15.8],  # 70-90% hum
#     ],
#         [  # 20 deg C
#             [0.1, 0.3, 0.6, 1.0, 1.9, 5.8, 20.3],  # 30-50% hum
#             [0.1, 0.3, 0.6, 1.0, 1.7, 4.1, 13.5],  # 50-70% hum
#             [0.1, 0.3, 0.6, 1.1, 1.7, 3.5, 10.6],  # 70-90% hum
#     ]
#     ]
#
#     # @staticmethod
#     # def getAttenuations(distance, temperature, relativeHumidity, units='dB'):
#     #     closestTemperatureIdx = np.argmin(
#     #         np.sqrt((np.array(AirAttenuationTable.temperatures) - temperature) ** 2))
#     #     closestHumidityIdx = np.argmin(
#     #         np.sqrt((np.array(AirAttenuationTable.relativeHumidities) - relativeHumidity) ** 2))
#     #
#     #     attenuations = np.array(
#     #         AirAttenuationTable.table[closestTemperatureIdx][closestHumidityIdx])
#     #     frequencies = np.array(AirAttenuationTable.frequencies)
#     #     eps = np.finfo(np.float).eps
#     #     attenuations = np.clip(distance * 1e-3 * attenuations, 0.0, 1.0 - eps)
#     #     if units == 'dB':
#     #         eps = np.finfo(np.float).eps
#     #         attenuations = 20.0 * np.log10(1.0 - attenuations + eps)
#     #     elif units == 'normalized':
#     #         # Nothing to do
#     #         pass
#     #     else:
#     #         raise Exception('Unsupported units: %s' % (units))
#     #     return attenuations, frequencies
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




class AirAttenuationTable(object):
    # From: Auralization : fundamentals of acoustics, modelling, simulation,
    # algorithms and acoustic virtual reality
    # Note: THe expansion is done merely for evaluation on the simulation with real collected data
    ##   which was collected under 19.5C temperature and 41.7% relative humidity. Further work can be done
    ##   in the future.

    ### The coefficients "alpha" from The calcualtor is in dB unit that is absorpted, which needs to be converted according to:
    # alpha = 20 log(P(absorpted)/P0)
    temperatures = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    relativeHumidities = [40.0, 42.0, 44.0, 46.0, 48.0, 50.0]
    frequencies = [20,     25.0,     31.5,    40.0,    50.0,    63.0,   80.0,   100.0,
                   125.0,  160.0,    200.0,   250.0,   315.0,   400.0,  500.0,  630.0,
                   800.0,  1000.0,   1250.0,  1600.0,  2000.0,  2500.0, 3150.0, 4000.0,
                   5000.0, 6300.0,   8000.0,  10000.0, 12500.0, 16000,  20000]

    # Air attenuation coefficient, in 10^-3 / m

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


class FilterBank(object):
    def __init__(self, n, centerFrequencies, samplingRate, maxLength=None):
        self.n = n
        if n % 2 == 0:
            self.n = n + 1
            logger.warn(
                'Length of the FIR filter adjusted to the next odd number to ensure symmetry: %d' % (self.n))
        else:
            self.n = n

        self.centerFrequencies = centerFrequencies
        self.samplingRate = samplingRate
        self.maxLength = maxLength

        centerFrequencies = np.array(centerFrequencies, dtype=np.float)
        centerNormFreqs = centerFrequencies / (self.samplingRate / 2.0)
        cutoffs = centerNormFreqs[:-1] + np.diff(centerNormFreqs) / 2


        filters = []
        for i in range(len(centerFrequencies)):
            if i == 0:
                # Low-pass filter
                b = signal.firwin(self.n, cutoff=cutoffs[0], window='hamming')
            elif i == len(centerFrequencies) - 1:
                # High-pass filter
                b = signal.firwin(
                    self.n, cutoff=cutoffs[-1], window='hamming', pass_zero=False)
            else:
                # Band-pass filter
                b = signal.firwin(
                    self.n, [cutoffs[i - 1], cutoffs[i]], pass_zero=False)

            filters.append(b)
        self.filters = np.array(filters)

        self._precomputeFiltersFourier()

    def _precomputeFiltersFourier(self):
        N = self.filters.shape[-1]
        if self.maxLength is not None:
            N = self.maxLength

        self.filtersFourier = np.fft.fft(self.filters, N)

    def getScaledImpulseResponse(self, scales=1):
        if not isinstance(scales, (list, tuple)):
            scales = scales * np.ones(len(self.filters))
        return np.sum(self.filters * scales[:, np.newaxis], axis=0)

    def getScaledImpulseResponseFourier(self, scales=1):
        if not isinstance(scales, (list, tuple)):
            scales = scales * np.ones(len(self.filters))
        return np.sum(self.filters * scales[:, np.newaxis], axis=0)

    def display(self, scales=1, merged=False):
        # Adapted from: http://mpastell.com/2010/01/18/fir-with-scipy/

        if merged:
            b = self.getScaledImpulseResponse(scales)
            filters = [b]
        else:
            filters = np.copy(self.filters)
            if not isinstance(scales, (list, tuple)):
                scales = scales * np.ones(len(filters))
            filters *= scales[:, np.newaxis]

        fig = plt.figure(figsize=(8, 6), facecolor='white', frameon=True)
        for b in filters:
            w, h = signal.freqz(b, 1)
            h_dB = 20 * np.log10(abs(h))
            plt.subplot(211)
            plt.plot(w / max(w), h_dB)
            plt.ylim(-150, 5)
            plt.ylabel('Magnitude (db)')
            plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
            plt.title(r'Frequency response')
            plt.subplot(212)
            h_Phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))
            plt.plot(w / max(w), h_Phase)
            plt.ylabel('Phase (radians)')
            plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
            plt.title(r'Phase response')
            plt.subplots_adjust(hspace=0.5)
        return fig


def getPathLength(path):
    # Calculate path length and corresponding delay
    pathLength = 0.0
    lastPt = path.m_points[0]
    for pt in path.m_points[1:]:
        pathLength += np.sqrt((lastPt.x - pt.x) ** 2 +
                              (lastPt.y - pt.y) ** 2 +
                              (lastPt.z - pt.z) ** 2)
        lastPt = pt

    # NOTE: EVERT works in milimeter units
    pathLength = pathLength / 1000.0  # mm to m
    return pathLength


def getIntersectionPointsFromPath(path):
    pts = []
    epts = path.m_points
    for i in range(len(epts)):
        if i > 0:
            segLength = np.sqrt((epts[i - 1].x - epts[i].x) ** 2 +
                                (epts[i - 1].y - epts[i].y) ** 2 +
                                (epts[i - 1].z - epts[i].z) ** 2)

            # Skip duplicated points
            # TODO: we may not need to check for duplicates in geometry, as
            # this was due to another bug
            if segLength == 0.0:
                continue

        # NOTE: EVERT works in milimeter units
        pts.append(LVector3f(epts[i].x / 1000.0,
                             epts[i].y / 1000.0, epts[i].z / 1000.0))
    return pts


def getIntersectedMaterialIdsFromPath(path):
    polygons = []
    lastPt = path.m_points[0]
    for i, pt in enumerate(path.m_points[1:]):
        segLength = np.sqrt((lastPt.x - pt.x) ** 2 +
                            (lastPt.y - pt.y) ** 2 +
                            (lastPt.z - pt.z) ** 2)

        # Skip duplicated points
        # TODO: we may not need to check for duplicates in geometry, as this
        # was due to another bug
        if segLength == 0.0:
            continue

        if i >= 2:
            polygons.append(path.m_polygons[i - 2])
        lastPt = pt

    materialIds = []
    for polygon in polygons:
        materialIds.append(polygon.getMaterialId())

    return materialIds


# Moller-Trumbore ray-triangle intersection algorithm vectorized with Numpy
# Adapted from: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
# See also:
# https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
def rayIntersectsTriangles(startPt, endPt, triangles, eps=1e-3):
    mask = np.ones((triangles.shape[0],), dtype=np.bool)

    d = np.linalg.norm((endPt - startPt), 2)
    vdir = (endPt - startPt) / d

    vertex0, vertex1, vertex2 = triangles[:, 0,
                                          :], triangles[:, 1, :], triangles[:, 2, :]
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0

    h = np.cross(vdir, edge2)
    a = np.einsum('ij,ij->i', edge1, h)
    mask &= np.logical_not((a > -eps) & (a < eps))

    f = 1 / a
    s = startPt - vertex0
    u = f * np.einsum('ij,ij->i', s, h)
    mask &= np.logical_not((u < 0.0) | (u > 1.0))

    q = np.cross(s, edge1)
    v = f * np.einsum('ij,ij->i', vdir[np.newaxis, :], q)
    mask &= np.logical_not((v < 0.0) | (u + v > 1.0))

    # NOTE: t is the distance from the start point to the intersection on the
    # plane defined by the triangle
    t = f * np.einsum('ij,ij->i', edge2, q)
    mask &= (t > eps)
    mask &= np.logical_not(np.isclose(t, d[np.newaxis], atol=eps))
    mask &= (t < d[np.newaxis])

    return np.any(mask)


def validatePath(path, triangles, eps):
    isValid = True
    pts = getIntersectionPointsFromPath(path)
    lastPt = pts[0]
    for pt in pts[1:]:

        if rayIntersectsTriangles(np.array([lastPt.x, lastPt.y, lastPt.z]),
                                  np.array([pt.x, pt.y, pt.z]),
                                  triangles,
                                  eps):
            isValid = False
            break

        lastPt = pt

    return isValid


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


class EvertAcoustics(World):
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

    def __init__(self, scene, hrtf=None, samplingRate=16000, maximumOrder=3, materialAbsorption=True,
                 frequencyDependent=True, debug=False,
                 microphoneTransform=None, objectMode='box', minWidthThresholdPolygons=0.0, maxImpulseLength=1.0,
                 maxBufferLength=3.0,
                 threshold=120.0, cameraMask=BitMask32.allOn()):

        super(EvertAcoustics, self).__init__()

        self.__dict__.update(scene=scene, hrtf=hrtf, samplingRate=samplingRate, maximumOrder=maximumOrder,
                             materialAbsorption=materialAbsorption,
                             frequencyDependent=frequencyDependent, debug=debug,
                             microphoneTransform=microphoneTransform, objectMode=objectMode,
                             minWidthThresholdPolygons=minWidthThresholdPolygons, maxImpulseLength=maxImpulseLength,
                             maxBufferLength=maxBufferLength, threshold=threshold, cameraMask=cameraMask)

        self.debug = debug
        self.samplingRate = samplingRate
        self.maximumOrder = maximumOrder
        self.materialAbsorption = materialAbsorption
        self.frequencyDependent = frequencyDependent

        self.world = EvertRoom()
        self.solutions = dict()
        self.render = NodePath('acoustic-render')

        self.sounds = dict()
        self.srcBuffers = dict()
        self.outBuffers = dict()
        self.ambientSounds = []

        self.filterbank = FilterBank(n=257,
                                     centerFrequencies=MaterialAbsorptionTable.frequencies,
                                     samplingRate=samplingRate)

        if self.hrtf is not None:
            self.hrtf.resample(samplingRate)

        self.setAirConditions()
        self.coefficientsForMaterialId = []

        self.agents = []
        self.sources = dict()
        self.acousticImpulseResponses = []

        self.initEvertDone = False

        if self.debug:
            self._preloadRayModels()

        self.scene.worlds['acoustics'] = self

        self.lastTaskTime = 0.0
        # taskMgr.add(self.update, 'acoustics', sort=0)

    def showRoomLayout(self,showCeilings=True,showWalls=True,showFloors=True):

        for np in self.scene.scene.findAllMatches('**/layouts/**/acoustics/*c'):
            if showCeilings:
                np.show(self.cameraMask)
                print("acoustics mask off")
            else:
                np.hide(BitMask32.allOn())
                print("acoustics mask on")

        for np in self.scene.scene.findAllMatches('**/layouts/**/acoustics/*w'):
            if showWalls:
                np.show(self.cameraMask)
            else:
                np.hide(BitMask32.allOn())

        for np in self.scene.scene.findAllMatches('**/layouts/**/acoustics/*f'):
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

    def visualizeEVERT(self):
        self._updateSources()
        self._updateListeners()
        viewer = EvertViewer(self.world, self.maximumOrder)
        viewer.show()

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

    def setAirConditions(self, pressureAtm=1.0, temperature=20.0, relativeHumidity=65.0):
        self.pressureAtm = pressureAtm
        self.temperature = temperature
        self.relativeHumidity = relativeHumidity

    def _calculateSoundSpeed(self):
        # Approximate speed of sound in dry (0% humidity) air, in meters per second, at temperatures near 0 deg C
        # TODO: alternative with humidity: http://resource.npl.co.uk/acoustics/techguides/speedair/
        # See paper: http://asa.scitation.org/doi/pdf/10.1121/1.405827
        return 331.3 * np.sqrt(1 + self.temperature / 273.15)

    def _calculateDelayAndAttenuation(self, path):

        pathLength = getPathLength(path)
        delay = pathLength / self._calculateSoundSpeed()

        # Calculate air attenuation coefficient (dB)
        airAttenuations, frequencies = AirAttenuationTable.getAttenuations(pathLength, self.temperature,
                                                                           self.relativeHumidity, units='dB')

        # Calculate spherical geometric spreading attenuation (dB)
        # TODO: should we make sure that sound is never amplified if too near?
        if pathLength > 0.0:
            distanceAttenuations = 20.0 * np.log10(1.0 / pathLength)
        else:
            distanceAttenuations = 0.0

        # Calculat material attenuation (dB)
        materialAttenuations = np.zeros(
            (len(MaterialAbsorptionTable.frequencies),))
        if self.materialAbsorption:
            for materialId in getIntersectedMaterialIdsFromPath(path):
                materialAbsorption = self.coefficientsForMaterialId[materialId]
                eps = np.finfo(np.float).eps
                materialAbsorptionDb = 20.0 * \
                    np.log10(1.0 - materialAbsorption + eps)
                materialAttenuations += materialAbsorptionDb

        # Total attenuation (dB)
        attenuation = airAttenuations + distanceAttenuations + materialAttenuations
        # assert np.all(attenuation < 0.0)

        return delay, attenuation, frequencies

    def _preloadRayModels(self, initialCacheSize=256):

        self.rays = []
        self.nbUsedRays = 0

        # Load cylinder model and calculate the scaling factor to make the
        # model unit-norm over the Z axis
        model = loadModel(os.path.join(MODEL_DATA_DIR, 'cylinder.egg'))
        minBounds, maxBounds = model.getTightBounds()
        dims = maxBounds - minBounds
        self.rayModelZScaling = 1.0 / dims.z
        model.removeNode()

        self.rayGroupNode = self.scene.scene.attachNewNode('rays')

        self._resizeRayCache(initialCacheSize)

    def _resizeRayCache(self, size):

        if size < len(self.rays):
            # Remove unused rays from the cache
            for model in self.rays[size:]:
                model.detachNode()
                model.removeNode()

            self.rays = self.rays[:size]
        else:
            # Add new rays to the cache
            for _ in range(size - len(self.rays)):
                # Load cylinder model
                model = loadModel(os.path.join(MODEL_DATA_DIR, 'cylinder.egg'))
                model.clearMaterial()
                model.reparentTo(self.rayGroupNode)
                model.hide(BitMask32.allOn())

                self.rays.append(model)

    def _updateRayModelFromEndpoints(self, model, startPt, endPt, radius, color):

        # Clear previous transform
        model.clearMat()

        # Reference length and center position of ray
        refLength = np.sqrt((endPt.x - startPt.x) ** 2 +
                            (endPt.y - startPt.y) ** 2 +
                            (endPt.z - startPt.z) ** 2)
        refCenter = (endPt + startPt) / 2.0

        # Change orientation by calculating rotation angles from the endpoints
        # NOTE: H angle is how the model rotates around the (0, 0, 1) axis,
        #       P angle how much it rotates around the (1, 0, 0) axis,
        #       R angle how much it rotates around the (0, 1, 0) axis.
        normVec = (endPt - startPt) / refLength

        # Rotation around X in the Y-Z plane
        angle = np.pi / 2 + np.arcsin(-normVec.z)
        model.setHpr(0.0, angle * 180 / np.pi, 0.0)
        transAroundX = model.getMat()

        # Rotation around Z in the X-Y plane
        angle = np.pi / 2 + np.arctan2(normVec.y, normVec.x)
        model.setHpr(angle * 180 / np.pi, 0.0, 0.0)
        transAroundZ = model.getMat()

        model.setMat(transAroundX * transAroundZ)

        # Move ray to reference center position
        minBounds, maxBounds = model.getTightBounds()
        center = minBounds + (maxBounds - minBounds) / 2.0
        deltaCenter = center - model.getPos()
        model.setPos(refCenter - deltaCenter)

        # Change radius based on attenuation
        model.setScale((radius, radius, refLength * self.rayModelZScaling))

        # Change color
        model.setColor(*color)

    def _calculateAttenuationPerSegment(self, path):

        pts = getIntersectionPointsFromPath(path)

        totalSegAttenuations = np.zeros((len(pts) - 1,))
        lastPt = pts[0]
        cumLength = 0.0
        for i, pt in enumerate(pts[1:]):
            segLength = np.sqrt((lastPt.x - pt.x) ** 2 +
                                (lastPt.y - pt.y) ** 2 +
                                (lastPt.z - pt.z) ** 2)
            assert segLength > 0.0
            cumLength += segLength

            # Calculate air attenuation coefficient (dB)
            airAttenuations, _ = AirAttenuationTable.getAttenuations(cumLength, self.temperature, self.relativeHumidity,
                                                                     units='dB')

            # Calculate spherical geometric spreading attenuation (dB)
            distanceAttenuations = 20.0 * np.log10(1.0 / cumLength)

            totalSegAttenuations[i] = np.mean(
                airAttenuations) + distanceAttenuations

            lastPt = pt

        # Calculate material attenuation (dB)
        if self.materialAbsorption:
            for i, materialId in enumerate(getIntersectedMaterialIdsFromPath(path)):
                materialAbsorption = self.coefficientsForMaterialId[materialId]
                eps = np.finfo(np.float).eps
                materialAbsorptionDb = 20.0 * \
                    np.log10(1.0 - materialAbsorption + eps)
                totalSegAttenuations[i + 1:] += np.mean(materialAbsorptionDb)

        # assert np.all(totalSegAttenuations < 0.0)

        return totalSegAttenuations

    def _renderAcousticPath(self, path, color):

        totalSegAttenuationsDb = self._calculateAttenuationPerSegment(path)

        pts = getIntersectionPointsFromPath(path)
        startPt = pts[0]
        for endPt, attenuationDb in zip(pts[1:], totalSegAttenuationsDb):

            coefficient = 10.0 ** (attenuationDb / 20.0)
            # assert coefficient >= 0.0 and coefficient <= 1.0
            radius = np.clip(self.maxRayRadius * coefficient,
                             a_min=self.minRayRadius, a_max=self.maxRayRadius)

            if self.nbUsedRays < len(self.rays):
                model = self.rays[self.nbUsedRays]
                self._updateRayModelFromEndpoints(
                    model, startPt, endPt, radius, color)
                model.show(self.cameraMask)
                self.nbUsedRays += 1
            else:
                nextSize = 2 * len(self.rays)
                logger.debug('Ray cache is full: increasing the size from %d to %d' % (
                    len(self.rays), nextSize))
                self._resizeRayCache(nextSize)

            startPt = endPt

    def _updateRenderedAcousticSolutions(self):

        # Reset the number of used rays
        self.nbUsedRays = 0

        # Loop for all agent
        # Draw each solution with a different color
        i = 0
        for agent in self.agents:

            # Loop for all sounds
            if agent.getName() in self.solutions:
                for solution, _, _ in self.solutions[agent.getName()]:

                    # Rotate amongst colors of the predefined table
                    color = self.rayColors[i % len(self.rayColors)]
                    i += 1

                    # Sort by increasing path lengh
                    paths = []
                    for n in range(solution.numPaths()):
                        path = solution.getPath(n)
                        pathLength = getPathLength(path)
                        paths.append((pathLength, path))
                    paths.sort(key=lambda x: x[0])
                    paths = [path for _, path in paths]
                    logger.debug('Number of paths found for solution %d: %d' % (
                        i, solution.numPaths()))

                    # Draw each solution path
                    for path in paths:
                        self._renderAcousticPath(path, color)

        # Hide all unused rays in the cache
        for i in range(self.nbUsedRays, len(self.rays)):
            self.rays[i].hide(BitMask32.allOn())

    def _calculatePathRelativeToMicrophone(self, path, microphoneNp):

        # Get the last segment of the path
        fromPt = LVecBase3(
            path.m_points[-2].x, path.m_points[-2].y, path.m_points[-2].z) / 1000.0  # mm to m
        toPt = LVecBase3(
            path.m_points[-1].x, path.m_points[-1].y, path.m_points[-1].z) / 1000.0  # mm to m
        assert np.allclose(vec3ToNumpyArray(toPt),
                           vec3ToNumpyArray(
                               microphoneNp.getNetTransform().getPos()),
                           atol=1e-6)

        srcDirVec = (fromPt - toPt).normalized()

        headTransform = microphoneNp.getNetTransform()
        headDirVec = headTransform.getNormQuat().getForward(CS_zup_right)

        # XXX: apply a more general calculation of azimut and elevation angles
        headRollAngle = headTransform.getHpr().getZ()
        if not np.allclose(headRollAngle, 0.0, atol=1e-6):
            logger.warn(
                'Microphone has non-zero roll angle, which is not taken into account!')

        # Get the azimut in X-Y plane
        srcDirVecXY = LVector3f(srcDirVec.x, srcDirVec.y, 0.0).normalized()
        headDirVecXY = LVector3f(headDirVec.x, headDirVec.y, 0.0).normalized()
        azimut = srcDirVecXY.signedAngleRad(
            headDirVecXY, LVector3f(0.0, 0.0, 1.0))

        # Get the elevation in Y-Z plane
        srcDirVecYZ = LVector3f(0.0, srcDirVec.y, srcDirVec.z).normalized()
        headDirVecYZ = LVector3f(0.0, headDirVec.y, headDirVec.z).normalized()
        elevation = srcDirVecYZ.signedAngleRad(
            headDirVecYZ, LVector3f(-1.0, 0.0, 0.0))

        return azimut, elevation

    def calculateImpulseResponse(self, srcName, lstName):

        solution = None
        for sol, _, solSrcName in self.solutions[lstName]:
            if solSrcName == srcName:
                solution = sol
                break
        assert solution is not None

        impulse = self._calculateImpulseResponse(solution, lstName)


        agentNp = self.scene.scene.find('**/' + lstName)
        microphoneNp = agentNp.find('**/acoustics/microphone*')
        sourceNp = self.scene.scene.find('**/' + srcName)

        return AcousticImpulseResponse(impulse, self.samplingRate, sourceNp, microphoneNp)

    def _calculateImpulseResponse(self, solution, lstName):

        # Calculate impulse response in time domain, accounting for frequency-dependent absorption,
        # then when it is time to convolve with the HRTF, do it in fourrier
        # domain.

        # TODO: Get the source and microphone related to this solution
        agentNp = self.scene.scene.find('**/' + lstName)
        microphoneNp = agentNp.find('**/acoustics/microphone*')


        if self.hrtf is not None:
            nbChannels = len(self.hrtf.channels)
        else:
            nbChannels = 1

        impulse = np.zeros(
            (nbChannels, int(self.maxImpulseLength * self.samplingRate)))
        realImpulseLength = 0
        for i in range(solution.numPaths()):
            path = solution.getPath(i)

            delay, attenuationsDb, _ = self._calculateDelayAndAttenuation(path)

            # Random phase inversion
            phase = 1.0
            if np.random.random() > 0.5:
                phase *= -1

            # Add path impulse to global impulse
            delaySamples = int(delay * self.samplingRate)

            # Skip paths that are below attenuation threshold (dB)
            if np.any(abs(attenuationsDb) < self.threshold):

                if self.hrtf is not None:
                    # Calculate azimut and elevation angles compared to the
                    # agent
                    azimut, elevation = self._calculatePathRelativeToMicrophone(
                        path, microphoneNp)
                    hrtfImpulse = self.hrtf.getImpulseResponse(
                        azimut, elevation)

                if self.frequencyDependent:

                    # Skip paths that would have their impulse responses
                    # truncated at the end
                    if delaySamples + self.filterbank.n < impulse.shape[-1]:

                        linearGains = 10.0 ** (attenuationsDb / 20.0)
                        pathImpulse = self.filterbank.getScaledImpulseResponse(
                            linearGains)

                        for channel in range(nbChannels):

                            if self.hrtf is not None:
                                # FIXME: should be using 'full' mode for
                                # convolution, and flip the hrtf impulse?
                                pathImpulseChan = signal.fftconvolve(
                                    pathImpulse, hrtfImpulse[channel], mode='same')
                            else:
                                pathImpulseChan = pathImpulse

                            startIdx = int(
                                delaySamples - self.filterbank.n / 2)
                            endIdx = startIdx + len(pathImpulseChan) - 1
                            if startIdx < 0:
                                trimStartIdx = -startIdx
                                startIdx = 0
                            else:
                                trimStartIdx = 0

                            impulse[channel, startIdx:endIdx +
                                    1] += phase * pathImpulseChan[trimStartIdx:]

                        if endIdx + 1 > realImpulseLength:
                            realImpulseLength = endIdx + 1
                else:
                    # Use attenuation at 1000 Hz
                    linearGain = 10.0 ** (attenuationsDb[3] / 20.0)

                    for channel in range(nbChannels):

                        if self.hrtf is not None:
                            pathImpulseChan = linearGain * hrtfImpulse[channel]

                            # FIXME: should be checking for truncation at the
                            # beginning and end
                            startIdx = delaySamples
                            endIdx = startIdx + len(pathImpulseChan) - 1

                            impulse[channel, startIdx:endIdx +
                                    1] += phase * pathImpulseChan
                            if endIdx + 1 > realImpulseLength:
                                realImpulseLength = endIdx + 1

                        else:
                            impulse[channel, delaySamples] += phase * \
                                linearGain
                            if delaySamples + 1 > realImpulseLength:
                                realImpulseLength = delaySamples + 1

        # Trim impulse to effective length
        impulse = impulse[:, :realImpulseLength]

        return impulse

    def _initLayoutModels(self):

        # Load layout objects as meshes
        # Poosibly can put in the object as an obstacle
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

            self._addModelGeometry(acousticModel)

    def _initAgents(self):

        # Load agents
        for agent in self.scene.scene.findAllMatches('**/agents/agent*'):

            agent.setTag('acoustics-mode', 'listener')

            acousticNode = agent.attachNewNode('acoustics')

            # Load model for the microphone
            microphoneNp = acousticNode.attachNewNode('microphone')
            if self.microphoneTransform is not None:
                microphoneNp.setTransform(self.microphoneTransform)
            model = self._loadSphereModel(refCenter=LVecBase3(
                0.0, 0.0, 0.0), radius=0.15, color=(1.0, 0.0, 0.0, 1.0))
            model.reparentTo(microphoneNp)

            # Add listeners for EVERT
            lst = Listener()
            lst.setName(agent.getName())

            netMat = microphoneNp.getNetTransform().getMat()
            lstNetPos = netMat.getRow3(3)
            lstNetMat = netMat.getUpper3()
            lst.setPosition(Vector3(
                lstNetPos.x * 1000.0, lstNetPos.y * 1000.0, lstNetPos.z * 1000.0))  # m to mm

            # NOTE Orientation matrix: first column is direction vector, second
            # column is up vector, third column is right vector
            lst.setOrientation(Matrix3(lstNetMat.getCell(0, 0), lstNetMat.getCell(2, 0), -lstNetMat.getCell(1, 0),
                                       lstNetMat.getCell(0, 1), lstNetMat.getCell(
                                           2, 1), -lstNetMat.getCell(1, 1),
                                       lstNetMat.getCell(0, 2), lstNetMat.getCell(2, 2), -lstNetMat.getCell(1, 2)))

            self.world.addListener(lst)

            logger.debug('Agent %s: microphone at position (x=%f, y=%f, z=%f)' % (
                agent.getName(), lstNetPos.x, lstNetPos.y, lstNetPos.z))

            # Allocate output buffer
            self.outBuffers[agent.getName()] = np.array(
                [[] for _ in range(self.getNbOutputChannels())])

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

        materialId = int(model.getTag('materialId'))

        # Add polygons to EVERT engine
        # TODO: for bounding box approximations, we could reduce the number of triangles by
        # half if each face of the box was modelled as a single rectangular
        # polygon.
        polygons, triangles = getAcousticPolygonsFromModel(model)
        for polygon, triangle in zip(polygons, triangles):

            # Validate triangle (ignore if area is too small)
            dims = np.max(triangle, axis=0) - np.min(triangle, axis=0)
            s = np.sum(
                np.array(dims > self.minWidthThresholdPolygons, dtype=np.int))
            if s < 2:
                continue

            polygon.setMaterialId(materialId)
            self.world.addPolygon(polygon, Vector3(1.0, 1.0, 1.0))

    def _updateListeners(self):

        # Update positions of listeners
        for agent in self.agents:

            lst = None
            for i in range(self.world.numListeners()):
                listener = self.world.getListener(i)
                if listener.getName() == str(agent.getName()):
                    lst = listener
                    break
            assert lst is not None

            microphoneNp = agent.find('**/microphone*')
            netMat = microphoneNp.getNetTransform().getMat()
            lstNetPos = netMat.getRow3(3)
            lstNetMat = netMat.getUpper3()
            lst.setPosition(Vector3(
                lstNetPos.x * 1000.0, lstNetPos.y * 1000.0, lstNetPos.z * 1000.0))  # m to mm
            # NOTE Orientation matrix: first column is direction vector, second
            # column is up vector, third column is right vector
            lst.setOrientation(Matrix3(lstNetMat.getCell(0, 0), lstNetMat.getCell(2, 0), -lstNetMat.getCell(1, 0),
                                       lstNetMat.getCell(0, 1), lstNetMat.getCell(
                                           2, 1), -lstNetMat.getCell(1, 1),
                                       lstNetMat.getCell(0, 2), lstNetMat.getCell(2, 2), -lstNetMat.getCell(1, 2)))

            logger.debug('Agent %s: microphone at position (x=%f, y=%f, z=%f)' % (
                agent.getName(), lstNetPos.x, lstNetPos.y, lstNetPos.z))

    def _updateSources(self):

        # NOTE: no need to update position of source, since they are static
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

        if not self.initEvertDone:

            self._initLayoutModels()
            self._initObjects()
            self._initAgents()

            self.world.constructBSP() # not in use

            self._initPathSolutions()
            self.initEvertDone = True

        self._updateSources()
        self._updateListeners()
        self._updateSrcBuffers(dt)
        self._updateOutputBuffers(dt)

        if self.debug:
            self._updateRenderedAcousticSolutions()

    def update(self, task):

        # Get dt
        dt = task.time - self.lastTaskTime
        # dt2 = ClockObject.getGlobalClock().getDt()
        self.lastTaskTime = task.time
        assert dt >= 0.0

        if not self.initEvertDone:

            self._initLayoutModels()
            self._initObjects()
            self._initAgents()

            self.world.constructBSP()

            self._initPathSolutions()
            self.initEvertDone = True

        self._updateSources()
        self._updateListeners()
        self._updateSrcBuffers(dt)
        self._updateOutputBuffers(dt)

        if self.debug:
            self._updateRenderedAcousticSolutions()

        ClockObject.getGlobalClock().tick()

        return task.cont

    def getObservationsForAgent(self, name, clearBuffer=True):

        if self.hrtf is not None:
            channelNames = self.hrtf.channels
        else:
            channelNames = ['0']

        observations = dict()
        for i, channelName in enumerate(channelNames):
            observations['audio-buffer-%s' %
                         (channelName)] = self.outBuffers[name][i, :]

        if clearBuffer:
            # Clear output buffer
            self.outBuffers[name] = np.array(
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

        self.sounds[obj.getName()] = sound

        # Allocate a buffer for that sound
        nbMaxSamples = int(self.maxBufferLength * self.samplingRate)
        self.srcBuffers[sound] = np.zeros((nbMaxSamples,), np.float32)

        # Create source in EVERT
        src = Source()
        src.setName(obj.getName())

        # Get position and orientation of the static sound source
        sourceNetTrans = obj.getNetTransform().getMat()
        sourceNetPos = sourceNetTrans.getRow3(3)
        sourceNetMat = sourceNetTrans.getUpper3()
        src.setPosition(
            Vector3(sourceNetPos.x * 1000.0, sourceNetPos.y * 1000.0, sourceNetPos.z * 1000.0))  # m to mm

        # NOTE Orientation matrix: first column is direction vector, second
        # column is up vector, third column is right vector
        src.setOrientation(Matrix3(sourceNetMat.getCell(0, 0), sourceNetMat.getCell(2, 0), -sourceNetMat.getCell(1, 0),
                                   sourceNetMat.getCell(0, 1), sourceNetMat.getCell(
                                       2, 1), -sourceNetMat.getCell(1, 1),
                                   sourceNetMat.getCell(0, 2), sourceNetMat.getCell(2, 2), -sourceNetMat.getCell(1, 2)))

        logger.debug('Static source %s: at position (x=%f, y=%f, z=%f)' % (
            obj.getName(), sourceNetPos.x, sourceNetPos.y, sourceNetPos.z))

        # Add as source in EVERT
        self.world.addSource(src)

        # Add acoustic node in scene graph
        obj.setTag('acoustics-mode', 'source')
        acousticsNp = obj.attachNewNode('acoustics')
        if self.debug:
            # Load model for the sound source
            acousticModel = self._loadSphereModel(
                LVector3f(0, 0, 0), radius=0.15, color=(1.0, 0.0, 0.0, 1.0))
            acousticModel.reparentTo(acousticsNp)

        return 1

    def _initPathSolutions(self):

        # Create new solutions for all pairs of source-listener
        for i in range(self.world.numSources()):
            src = self.world.getSource(i)
            obj = self.scene.scene.find('**/objects/%s' % (src.getName()))
            sound = self.sounds[src.getName()]

            for l in range(self.world.numListeners()):
                lst = self.world.getListener(l)

                # WARNING: we have to create the PathSolution instances after adding all sources and listeners,
                # otherwise it leads to memory errors since sources can be
                # moved in memory-space.
                solution = PathSolution(
                    self.world, src, lst, self.maximumOrder)

                # Force update to pre-compute cache
                solution.update()

                if lst.getName() not in self.solutions:
                    self.solutions[lst.getName()] = []

                self.solutions[lst.getName()].append(
                    (solution, sound, obj.getName()))

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
                        data = sound.data[startIdx:endIdx + 1]
                    else:
                        # Get the sound data for this timestep
                        startIdx = int(tStart * sound.samplingRate)
                        data = sound.data[startIdx:]

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

                # Loop for all attached sounds
                if agent.getName() in self.solutions:
                    for solution, sound, _ in self.solutions[agent.getName()]:

                        # NOTE: fast update of the solution, since the beam
                        # tree should already has been pre-computed
                        solution.update()

                        # Get the impulse response related to this
                        # source-listener pair
                        impulse = self._calculateImpulseResponse(
                            solution, agent.getName())

                        if impulse.shape[-1] > 0:

                            # Get the source buffer related to the sound
                            buf = self.srcBuffers[sound]

                            # Convolve source with impulse and add contribution
                            # to output buffer
                            nbChannels = impulse.shape[0]
                            nbInSamples = newDataLength + impulse.shape[1] - 1
                            for channel in range(nbChannels):
                                outBuf[channel, :] += signal.convolve(
                                    buf[len(buf) - nbInSamples:], impulse[channel], mode='valid') * sound.getVolume()

                # Loop for all ambient sounds
                for sound in self.ambientSounds:

                    # Get the source buffer related to the sound
                    buf = self.srcBuffers[sound]

                    # Add contribution equally to all channels
                    nbChannels = self.getNbOutputChannels()
                    nbInSamples = newDataLength
                    for channel in range(nbChannels):
                        outBuf[channel,
                               :] += buf[len(buf) - nbInSamples:] * sound.getVolume()

                # XXX: should we concatenate to create a continuous buffer or
                # simply append as frames to a list?
                self.outBuffers[agent.getName()] = np.concatenate(
                    (self.outBuffers[agent.getName()], outBuf), axis=-1)
class EvertAudioSound(object):
    # Python implementation of AudioSound abstract class:
    # https://www.panda3d.org/reference/1.9.4/python/panda3d.core.AudioSound

    def __init__(self, filename):
        self.name = os.path.basename(filename)
        self.filename = filename

        # Load sound from file
        data, fs = sf.read(filename)

        # Make sure the sound is mono, and keep the first channel only
        if data.ndim == 2 and np.maximum(data.shape[0], data.shape[0]) >= 1:
            if data.shape[0]>data.shape[1]:
                data=np.transpose(data)
                data = np.sum(data,axis=0)
            else:
                data = np.sum(data,axis=0)

        # Normalize in the interval [-1, 1]
        data = data / np.max(np.abs(data))
        self.data =data
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

    def __init__(self, scene, hrtf=None, samplingRate=16000, maximumOrder=3, materialAbsorption=True,
                 frequencyDependent=True, debug=False,
                 microphoneTransform=None, objectMode='box', minWidthThresholdPolygons=0.0, maxImpulseLength=1.0,
                 maxBufferLength=3.0,
                 threshold=120.0, cameraMask=BitMask32.allOn()):

        super(EvertAcoustics_jz, self).__init__()

        self.__dict__.update(scene=scene, hrtf=hrtf, samplingRate=samplingRate, maximumOrder=maximumOrder,
                             materialAbsorption=materialAbsorption,
                             frequencyDependent=frequencyDependent, debug=debug,
                             microphoneTransform=microphoneTransform, objectMode=objectMode,
                             minWidthThresholdPolygons=minWidthThresholdPolygons, maxImpulseLength=maxImpulseLength,
                             maxBufferLength=maxBufferLength, threshold=threshold, cameraMask=cameraMask)

        self.debug = debug
        self.samplingRate = samplingRate
        self.maximumOrder = maximumOrder
        self.materialAbsorption = materialAbsorption
        self.frequencyDependent = frequencyDependent

        self.world = EvertRoom()
        self.solutions = dict()
        self.render = NodePath('acoustic-render')

        self.sounds = dict()
        self.srcBuffers = dict()
        self.outBuffers = dict()
        self.ambientSounds = []

        self.filterbank = FilterBank(n=257,
                                     centerFrequencies=MaterialAbsorptionTable.frequencies,
                                     samplingRate=samplingRate)

        if self.hrtf is not None:
            self.hrtf.resample(samplingRate)
        self.nbMicrophones = len(microphoneTransform)

        self.setAirConditions()
        self.coefficientsForMaterialId = []

        self.agents = []
        self.sources = dict()
        self.acousticImpulseResponses = []

        self.initEvertDone = False

        if self.debug:
            self._preloadRayModels()

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

    def visualizeEVERT(self):
        self._updateSources()
        self._updateListeners()
        viewer = EvertViewer(self.world, self.maximumOrder)
        viewer.show()

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

    def _calculateSoundSpeed(self):
        # Approximate speed of sound in dry (0% humidity) air, in meters per second, at temperatures near 0 deg C
        # TODO: alternative with humidity: http://resource.npl.co.uk/acoustics/techguides/speedair/
        # See paper: http://asa.scitation.org/doi/pdf/10.1121/1.405827
        return 331.3 * np.sqrt(1 + self.temperature / 273.15)

    def _calculateDelayAndAttenuation(self, path):

        pathLength = getPathLength(path)
        delay = pathLength / self._calculateSoundSpeed()

        # Calculate air attenuation coefficient (dB)
        airAttenuations, frequencies = AirAttenuationTable.getAttenuations(pathLength, self.temperature,
                                                                           self.relativeHumidity, units='dB')

        # Calculate spherical geometric spreading attenuation (dB)
        # TODO: should we make sure that sound is never amplified if too near?
        if pathLength > 0.0:
            distanceAttenuations = 20.0 * np.log10(1.0 / pathLength)
        else:
            distanceAttenuations = 0.0

        # Calculat material attenuation (dB)
        materialAttenuations = np.zeros(
            (len(MaterialAbsorptionTable.frequencies),))
        if self.materialAbsorption:
            for materialId in getIntersectedMaterialIdsFromPath(path):
                materialAbsorption = self.coefficientsForMaterialId[materialId]
                eps = np.finfo(np.float).eps
                materialAbsorptionDb = 20.0 * \
                    np.log10(1.0 - materialAbsorption + eps)
                materialAttenuations += materialAbsorptionDb

        # Total attenuation (dB)
        attenuation = airAttenuations + distanceAttenuations + materialAttenuations
        # assert np.all(attenuation < 0.0)

        return delay, attenuation, frequencies

    def _preloadRayModels(self, initialCacheSize=256):

        self.rays = []
        self.nbUsedRays = 0

        # Load cylinder model and calculate the scaling factor to make the
        # model unit-norm over the Z axis
        model = loadModel(os.path.join(MODEL_DATA_DIR, 'cylinder.egg'))
        minBounds, maxBounds = model.getTightBounds()
        dims = maxBounds - minBounds
        self.rayModelZScaling = 1.0 / dims.z
        model.removeNode()

        self.rayGroupNode = self.scene.scene.attachNewNode('rays')

        self._resizeRayCache(initialCacheSize)

    def _resizeRayCache(self, size):

        if size < len(self.rays):
            # Remove unused rays from the cache
            for model in self.rays[size:]:
                model.detachNode()
                model.removeNode()

            self.rays = self.rays[:size]
        else:
            # Add new rays to the cache
            for _ in range(size - len(self.rays)):
                # Load cylinder model
                model = loadModel(os.path.join(MODEL_DATA_DIR, 'cylinder.egg'))
                model.clearMaterial()
                model.reparentTo(self.rayGroupNode)
                model.hide(BitMask32.allOn())

                self.rays.append(model)

    def _updateRayModelFromEndpoints(self, model, startPt, endPt, radius, color):

        # Clear previous transform
        model.clearMat()

        # Reference length and center position of ray
        refLength = np.sqrt((endPt.x - startPt.x) ** 2 +
                            (endPt.y - startPt.y) ** 2 +
                            (endPt.z - startPt.z) ** 2)
        refCenter = (endPt + startPt) / 2.0

        # Change orientation by calculating rotation angles from the endpoints
        # NOTE: H angle is how the model rotates around the (0, 0, 1) axis,
        #       P angle how much it rotates around the (1, 0, 0) axis,
        #       R angle how much it rotates around the (0, 1, 0) axis.
        normVec = (endPt - startPt) / refLength

        # Rotation around X in the Y-Z plane
        angle = np.pi / 2 + np.arcsin(-normVec.z)
        model.setHpr(0.0, angle * 180 / np.pi, 0.0)
        transAroundX = model.getMat()

        # Rotation around Z in the X-Y plane
        angle = np.pi / 2 + np.arctan2(normVec.y, normVec.x)
        model.setHpr(angle * 180 / np.pi, 0.0, 0.0)
        transAroundZ = model.getMat()

        model.setMat(transAroundX * transAroundZ)

        # Move ray to reference center position
        minBounds, maxBounds = model.getTightBounds()
        center = minBounds + (maxBounds - minBounds) / 2.0
        deltaCenter = center - model.getPos()
        model.setPos(refCenter - deltaCenter)

        # Change radius based on attenuation
        model.setScale((radius, radius, refLength * self.rayModelZScaling))

        # Change color
        model.setColor(*color)

    def _calculateAttenuationPerSegment(self, path):

        pts = getIntersectionPointsFromPath(path)

        totalSegAttenuations = np.zeros((len(pts) - 1,))
        lastPt = pts[0]
        cumLength = 0.0
        for i, pt in enumerate(pts[1:]):
            segLength = np.sqrt((lastPt.x - pt.x) ** 2 +
                                (lastPt.y - pt.y) ** 2 +
                                (lastPt.z - pt.z) ** 2)
            assert segLength > 0.0
            cumLength += segLength
            # Calculate air attenuation coefficient (dB)
            airAttenuations, _ = AirAttenuationTable.getAttenuations(cumLength, self.temperature, self.relativeHumidity,
                                                                     units='dB')

            # Calculate spherical geometric spreading attenuation (dB)
            distanceAttenuations = 20.0 * np.log10(1.0 / cumLength)

            totalSegAttenuations[i] = np.mean(
                airAttenuations) + distanceAttenuations

            lastPt = pt

        # Calculate material attenuation (dB)
        if self.materialAbsorption:
            for i, materialId in enumerate(getIntersectedMaterialIdsFromPath(path)):
                materialAbsorption = self.coefficientsForMaterialId[materialId]
                eps = np.finfo(np.float).eps
                materialAbsorptionDb = 20.0 * \
                    np.log10(1.0 - materialAbsorption + eps)
                totalSegAttenuations[i + 1:] += np.mean(materialAbsorptionDb)

        # assert np.all(totalSegAttenuations < 0.0)

        return totalSegAttenuations

    def _renderAcousticPath(self, path, color):

        totalSegAttenuationsDb = self._calculateAttenuationPerSegment(path)

        pts = getIntersectionPointsFromPath(path)
        startPt = pts[0]
        for endPt, attenuationDb in zip(pts[1:], totalSegAttenuationsDb):

            coefficient = 10.0 ** (attenuationDb / 20.0)
            # assert coefficient >= 0.0 and coefficient <= 1.0
            radius = np.clip(self.maxRayRadius * coefficient,
                             a_min=self.minRayRadius, a_max=self.maxRayRadius)

            if self.nbUsedRays < len(self.rays):
                model = self.rays[self.nbUsedRays]
                self._updateRayModelFromEndpoints(
                    model, startPt, endPt, radius, color)
                model.show(self.cameraMask)
                self.nbUsedRays += 1
            else:
                nextSize = 2 * len(self.rays)
                logger.debug('Ray cache is full: increasing the size from %d to %d' % (
                    len(self.rays), nextSize))
                self._resizeRayCache(nextSize)

            startPt = endPt

    def _updateRenderedAcousticSolutions(self):

        # Reset the number of used rays
        self.nbUsedRays = 0

        # Loop for all agent
        # Draw each solution with a different color
        i = 0
        for agent in self.agents:
            for j in range(self.nbMicrophones):
                micName = str(agent.getName()+'-mic'+str(j))
                if micName in self.solutions:
                    for solution, _, _ in self.solutions[micName]:
                        color = self.rayColors[i % len(self.rayColors)]
                        i += 1

                        # Sort by increasing path lengh
                        paths = []
                        for n in range(solution.numPaths()):
                            path = solution.getPath(n)
                            pathLength = getPathLength(path)
                            paths.append((pathLength, path))
                        paths.sort(key=lambda x: x[0])
                        paths = [path for _, path in paths]
                        logger.debug('Number of paths found for solution %d: %d' % (
                            i, solution.numPaths()))

                        # Draw each solution path
                        for path in paths:
                            self._renderAcousticPath(path, color)

        # Hide all unused rays in the cache
        for i in range(self.nbUsedRays, len(self.rays)):
            self.rays[i].hide(BitMask32.allOn())

    ### used when hrtf is not None
    def _calculatePathRelativeToMicrophone(self, path, microphoneNp):

        # Get the last segment of the path
        fromPt = LVecBase3(
            path.m_points[-2].x, path.m_points[-2].y, path.m_points[-2].z) / 1000.0  # mm to m
        toPt = LVecBase3(
            path.m_points[-1].x, path.m_points[-1].y, path.m_points[-1].z) / 1000.0  # mm to m
        assert np.allclose(vec3ToNumpyArray(toPt),
                           vec3ToNumpyArray(
                               microphoneNp.getNetTransform().getPos()),
                           atol=1e-6)

        srcDirVec = (fromPt - toPt).normalized()

        microTransform = microphoneNp.getNetTransform()
        microDirVec = microTransform.getNormQuat().getForward(CS_zup_right)

        # XXX: apply a more general calculation of azimut and elevation angles
        microRollAngle = microTransform.getHpr().getZ()
        if not np.allclose(microRollAngle, 0.0, atol=1e-6):
            logger.warn(
                'Microphone has non-zero roll angle, which is not taken into account!')

        # Get the azimut in X-Y plane
        srcDirVecXY = LVector3f(srcDirVec.x, srcDirVec.y, 0.0).normalized()
        microDirVecXY = LVector3f(microDirVec.x, microDirVec.y, 0.0).normalized()
        azimut = srcDirVecXY.signedAngleRad(
            microDirVecXY, LVector3f(0.0, 0.0, 1.0))

        # Get the elevation in Y-Z plane
        srcDirVecYZ = LVector3f(0.0, srcDirVec.y, srcDirVec.z).normalized()
        microDirVecYZ = LVector3f(0.0, microDirVec.y, microDirVec.z).normalized()
        elevation = srcDirVecYZ.signedAngleRad(
            microDirVecYZ, LVector3f(-1.0, 0.0, 0.0))

        return azimut, elevation

    def calculateImpulseResponse(self, srcName, agentName):

        solution = None
        impulse = dict()
        micNps = dict()

        agentNp = self.scene.scene.find('**/' + agentName)
        sourceNp = self.scene.scene.find('**/' + srcName)

        for i in range(self.nbMicrophones):
            micName = agentName + '-mic'+str(i)
            for sol, _, solSrcName in self.solutions[micName]:
                if solSrcName == srcName:
                    solution = sol
                    break
            assert solution is not None
            impulse[i] = self._calculateImpulseResponse(solution, micName)
            micNps[i] = agentNp.find('**/acoustics/microphoneNp-'+str(i))


        return AcousticImpulseResponse(impulse, self.samplingRate, sourceNp, micNps)


    def _calculateImpulseResponse(self, solution, micName):

        # Calculate impulse response in time domain, accounting for frequency-dependent absorption,
        # then when it is time to convolve with the HRTF, do it in fourrier
        # domain.

        # TODO: Get the source and microphone related to this solution
        # The naming example of varaiable micName: agent-0-mic1
        agentNp = self.scene.scene.find('**/'+micName[:7])
        microphoneNp = agentNp.find('**/acoustics/microphoneNp-'+micName[-1])

        if self.hrtf is not None:
            nbChannels = len(self.hrtf.channels)
        else:
            nbChannels = 1

        impulse = np.zeros(
            (nbChannels, int(self.maxImpulseLength * self.samplingRate)))
        realImpulseLength = 0
        for i in range(solution.numPaths()):
            path = solution.getPath(i)

            delay, attenuationsDb, _ = self._calculateDelayAndAttenuation(path)

            # Random phase inversion
            phase = (-1)**i
            # if np.random.random() > 0.5:
            #     phase *= -1

            # Add path impulse to global impulse
            delaySamples = int(delay * self.samplingRate)

            # Skip paths that are below attenuation threshold (dB)
            if np.any(abs(attenuationsDb) < self.threshold):

                if self.hrtf is not None:
                    # Calculate azimut and elevation angles compared to the
                    # agent
                    azimut, elevation = self._calculatePathRelativeToMicrophone(
                        path, microphoneNp)
                    hrtfImpulse = self.hrtf.getImpulseResponse(
                        azimut, elevation)

                if self.frequencyDependent:

                    # Skip paths that would have their impulse responses
                    # truncated at the end
                    if delaySamples + self.filterbank.n < impulse.shape[-1]:

                        linearGains = 10.0 ** (attenuationsDb / 20.0)
                        pathImpulse = self.filterbank.getScaledImpulseResponse(
                            linearGains)

                        for channel in range(nbChannels):

                            if self.hrtf is not None:
                                # FIXME: should be using 'full' mode for
                                # convolution, and flip the hrtf impulse?
                                pathImpulseChan = signal.fftconvolve(
                                    pathImpulse, hrtfImpulse[channel], mode='same')
                            else:
                                pathImpulseChan = pathImpulse

                            startIdx = int(
                                delaySamples - self.filterbank.n / 2)
                            endIdx = startIdx + len(pathImpulseChan) - 1
                            if startIdx < 0:
                                trimStartIdx = -startIdx
                                startIdx = 0
                            else:
                                trimStartIdx = 0

                            impulse[channel, startIdx:endIdx +
                                    1] += phase * pathImpulseChan[trimStartIdx:]

                        if endIdx + 1 > realImpulseLength:
                            realImpulseLength = endIdx + 1
                else:
                    # Use attenuation at 1000 Hz
                    linearGain = 10.0 ** (attenuationsDb[3] / 20.0)

                    for channel in range(nbChannels):

                        if self.hrtf is not None:
                            pathImpulseChan = linearGain * hrtfImpulse[channel]

                            # FIXME: should be checking for truncation at the
                            # beginning and end
                            startIdx = delaySamples
                            endIdx = startIdx + len(pathImpulseChan) - 1

                            impulse[channel, startIdx:endIdx +
                                    1] += phase * pathImpulseChan
                            if endIdx + 1 > realImpulseLength:
                                realImpulseLength = endIdx + 1

                        else:
                            impulse[channel, delaySamples] += phase * \
                                linearGain
                            if delaySamples + 1 > realImpulseLength:
                                realImpulseLength = delaySamples + 1

        # Trim impulse to effective length
        impulse = impulse[:, :realImpulseLength]

        return impulse

    def _initLayoutModels(self):

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

            self._addModelGeometry(acousticModel)

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

                    # Add listeners for EVERT
                    lst = Listener()
                    lst.setName(str(agent.getName()+'-mic'+str(i)))

                    netMat = MicNodeNp.getNetTransform().getMat()
                    lstNetPos = netMat.getRow3(3)
                    lstNetMat = netMat.getUpper3()
                    lst.setPosition(Vector3(
                        lstNetPos.x * 1000.0, lstNetPos.y * 1000.0, lstNetPos.z * 1000.0))  # m to mm

                    # NOTE Orientation matrix: first column is direction vector, second
                    # column is up vector, third column is right vector
                    lst.setOrientation(Matrix3(lstNetMat.getCell(0, 0), lstNetMat.getCell(2, 0), -lstNetMat.getCell(1, 0),
                                       lstNetMat.getCell(0, 1), lstNetMat.getCell(
                                           2, 1), -lstNetMat.getCell(1, 1),
                                       lstNetMat.getCell(0, 2), lstNetMat.getCell(2, 2), -lstNetMat.getCell(1, 2)))

                    self.world.addListener(lst)
                    # # Allocate output buffer
                    self.outBuffers[agent.getName()][lst.getName()] = np.array(
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

        materialId = int(model.getTag('materialId'))

        # Add polygons to EVERT engine
        # TODO: for bounding box approximations, we could reduce the number of triangles by
        # half if each face of the box was modelled as a single rectangular
        # polygon.
        polygons, triangles = getAcousticPolygonsFromModel(model)
        for polygon, triangle in zip(polygons, triangles):

            # Validate triangle (ignore if area is too small)
            dims = np.max(triangle, axis=0) - np.min(triangle, axis=0)
            s = np.sum(
                np.array(dims > self.minWidthThresholdPolygons, dtype=np.int))
            if s < 2:
                continue

            polygon.setMaterialId(materialId)
            self.world.addPolygon(polygon, Vector3(1.0, 1.0, 1.0))

    def _updateListeners(self):

        # Update positions of listeners

        for i in range(self.world.numListeners()):
            lst = None
            listener = self.world.getListener(i)
            for agent in self.agents:
                for j in range(self.nbMicrophones):

                    if listener.getName() == str(agent.getName()+'-mic'+str(j)) :
                        lst = listener
                        MicNodeName='microphoneNp-%s' % (j)
                        microphoneNp = agent.find('**/%s'% (MicNodeName))
                        break
                else:
                    continue
                break
            assert lst is not None
            netMat = microphoneNp.getNetTransform().getMat()
            lstNetPos = netMat.getRow3(3)
            lstNetMat = netMat.getUpper3()
            lst.setPosition(Vector3(
                lstNetPos.x * 1000.0, lstNetPos.y * 1000.0, lstNetPos.z * 1000.0))  # m to mm

            # NOTE Orientation matrix: first column is direction vector, second
            # column is up vector, third column is right vector
            lst.setOrientation(Matrix3(lstNetMat.getCell(0, 0), lstNetMat.getCell(2, 0), -lstNetMat.getCell(1, 0),
                                       lstNetMat.getCell(0, 1), lstNetMat.getCell(
                                           2, 1), -lstNetMat.getCell(1, 1),
                                       lstNetMat.getCell(0, 2), lstNetMat.getCell(2, 2), -lstNetMat.getCell(1, 2)))

            # logger.debug('Agent %s: microphone at position (x=%f, y=%f, z=%f)' % (
            #     agent.getName(), lstNetPos.x, lstNetPos.y, lstNetPos.z))

    def _updateSources(self):

        # NOTE: no need to update position of source, since they are static
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

        if not self.initEvertDone:

            self._initLayoutModels()
            self._initObjects()
            self._initAgents()

            self.world.constructBSP()

            self._initPathSolutions()
            self.initEvertDone = True

        self._updateSources()
        self._updateListeners()
        self._updateSrcBuffers(dt)
        self._updateOutputBuffers(dt)

        if self.debug:
            self._updateRenderedAcousticSolutions()

    def update(self, task):

        # Get dt
        dt = task.time - self.lastTaskTime
        # dt2 = ClockObject.getGlobalClock().getDt()
        self.lastTaskTime = task.time
        assert dt >= 0.0

        if not self.initEvertDone:

            self._initLayoutModels()
            self._initObjects()
            self._initAgents()

            self.world.constructBSP()

            self._initPathSolutions()
            self.initEvertDone = True

        self._updateSources()
        self._updateListeners()
        self._updateSrcBuffers(dt)
        self._updateOutputBuffers(dt)

        if self.debug:
            self._updateRenderedAcousticSolutions()

        ClockObject.getGlobalClock().tick()

        return task.cont

    def getObservationsForAgent(self, name, clearBuffer=True):

        observations = dict()
        for i in range(self.nbMicrophones):

            if self.hrtf is not None:
                channelNames = self.hrtf.channels
            else:
                channelNames = ['0']

            micName = name+'-mic'+str(i)
            observations[micName]=dict()

            for i, channelName in enumerate(channelNames):
                observations[micName]['audio-buffer-%s' %
                            (channelName)] = self.outBuffers[name][micName][i, :]

            if clearBuffer:
                # Clear output buffer
                self.outBuffers[name][micName] = np.array(
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

        self.sounds[obj.getName()] = sound

        # Allocate a buffer for that sound
        nbMaxSamples = int(self.maxBufferLength * self.samplingRate)
        self.srcBuffers[sound] = np.zeros((nbMaxSamples,), np.float32)

        # Create source in EVERT
        src = Source()
        src.setName(obj.getName())

        # Get position and orientation of the static sound source
        sourceNetTrans = obj.getNetTransform().getMat()
        sourceNetPos = sourceNetTrans.getRow3(3)
        sourceNetMat = sourceNetTrans.getUpper3()
        src.setPosition(
            Vector3(sourceNetPos.x * 1000.0, sourceNetPos.y * 1000.0, sourceNetPos.z * 1000.0))  # m to mm

        # NOTE Orientation matrix: first column is direction vector, second
        # column is up vector, third column is right vector
        src.setOrientation(Matrix3(sourceNetMat.getCell(0, 0), sourceNetMat.getCell(2, 0), -sourceNetMat.getCell(1, 0),
                                   sourceNetMat.getCell(0, 1), sourceNetMat.getCell(
                                       2, 1), -sourceNetMat.getCell(1, 1),
                                   sourceNetMat.getCell(0, 2), sourceNetMat.getCell(2, 2), -sourceNetMat.getCell(1, 2)))

        logger.debug('Static source %s: at position (x=%f, y=%f, z=%f)' % (
            obj.getName(), sourceNetPos.x, sourceNetPos.y, sourceNetPos.z))

        # Add as source in EVERT
        self.world.addSource(src)

        # Add acoustic node in scene graph
        obj.setTag('acoustics-mode', 'source')
        acousticsNp = obj.attachNewNode('acoustics')
        if self.debug:
            # Load model for the sound source
            acousticModel = self._loadSphereModel(
                LVector3f(0, 0, 0), radius=0.25, color=(1.0, 0.0, 0.0, 1.0))
            acousticModel.reparentTo(acousticsNp)

        return 1

    def _initPathSolutions(self):

        # Create new solutions for all pairs of source-listener
        for i in range(self.world.numSources()):
            src = self.world.getSource(i)
            obj = self.scene.scene.find('**/objects/%s' % (src.getName()))
            sound = self.sounds[src.getName()]

            for l in range(self.world.numListeners()):
                lst = self.world.getListener(l)

                # WARNING: we have to create the PathSolution instances after adding all sources and listeners,
                # otherwise it leads to memory errors since sources can be
                # moved in memory-space.
                solution = PathSolution(
                    self.world, src, lst, self.maximumOrder)

                # Force update to pre-compute cache
                solution.update()

                if lst.getName() not in self.solutions:
                    self.solutions[lst.getName()] = []

                self.solutions[lst.getName()].append(
                    (solution, sound, obj.getName()))

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
                        data = sound.data[startIdx:endIdx + 1]
                    else:
                        # Get the sound data for this timestep
                        startIdx = int(tStart * sound.samplingRate)
                        data = sound.data[startIdx:]

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

                # Loop for listeners(microphones)
                for j in range(self.nbMicrophones):
                    lstName = str(agent.getName()+'-mic'+str(j))
                    if lstName in self.solutions:

                        for solution, sound, _ in self.solutions[lstName]:

                            # NOTE: fast update of the solution, since the beam
                            # tree should already has been pre-computed
                            solution.update()

                            # Get the impulse response related to this
                            # source-listener pair
                            impulse = self._calculateImpulseResponse(
                                solution, lstName)
                            if impulse.shape[-1] > 0:

                                # Get the source buffer related to the sound
                                buf = self.srcBuffers[sound]

                                # Convolve source with impulse and add contribution
                                # to output buffer
                                nbChannels = impulse.shape[0]
                                nbInSamples = newDataLength + impulse.shape[1] - 1
                                for channel in range(nbChannels):
                                    outBuf[channel, :] += signal.convolve(
                                        buf[len(buf) - nbInSamples:], impulse[channel], mode='valid') * sound.getVolume()

                        # Loop for all ambient sounds
                        for sound in self.ambientSounds:

                        # Get the source buffer related to the sound
                            buf = self.srcBuffers[sound]

                            # Add contribution equally to all channels
                            nbChannels = self.getNbOutputChannels()
                            nbInSamples = newDataLength
                            for channel in range(nbChannels):
                                outBuf[channel,
                                    :] += buf[len(buf) - nbInSamples:] * sound.getVolume()

                    # XXX: should we concatenate to create a continuous buffer or
                    # simply append as frames to a list?
                    self.outBuffers[agent.getName()][lstName] = np.concatenate(
                        (self.outBuffers[agent.getName()][lstName], outBuf), axis=-1)


