import numpy as np
import time
import json
import random
import math
import copy
import matplotlib.pyplot as plt
from numba import jit


def load_json(path):
    """Load data from .json file and turn it into numpy array"""

    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)


def createWallGrid(Grid, Helpinggrid):
    """Return wall code, a number between -1 and 14 telling in which direction walls are adjacent"""

    Wallgrid = np.empty_like(Grid)

    for index, _ in np.ndenumerate(Grid):

        # check if grid cell adjoins a wall, convert to binary and then to int
        corner1 = Helpinggrid[index[0]][index[1]]
        corner2 = Helpinggrid[index[0]][index[1] + 1]
        corner3 = Helpinggrid[index[0] + 1][index[1]]
        corner4 = Helpinggrid[index[0] + 1][index[1] + 1]
        corners = [str(corner1), str(corner2), str(corner3), str(corner4)]

        binary = -1
        if "-1" not in corners:
            binary = "".join(corners)
            wallnum = int(binary, base=2)
        elif "-1" in corners:
            wallnum = -1
        # set same index as grid cell to corresponding wall number (can be from -1 to 15)
        Wallgrid[index[0]][index[1]] = wallnum

    return Wallgrid


@jit(nopython=True)
def createAdditionalBeams(wallgrid, Helpinggrid):
    """Place beams from a random corner point in a random direction."""

    chosenCornerPoints = []
    for index, val in np.ndenumerate(wallgrid):
        # pick a corner grid cell with 50% chance
        if wallgrid[index[0]][index[1]] in [1, 2, 4, 8] and random.randint(0, 9) < 9:
            chosenCornerPoints.append((index, val))
            print("Placing additional beams!")

    for index, val in chosenCornerPoints:
        beamDirection = random.randint(0, 2)
        # print("beam direction: ", beamDirection)
        if beamDirection == 1:
            if val == 1:
                Helpinggrid = placeBeam(Helpinggrid, index, 1, 0, (0, -1))
            elif val == 2:
                Helpinggrid = placeBeam(Helpinggrid, index, 1, 1, (0, 1))
            elif val == 4:
                Helpinggrid = placeBeam(Helpinggrid, index, 0, 0, (0, -1))
            elif val == 8:
                Helpinggrid = placeBeam(Helpinggrid, index, 0, 1, (0, 1))
        elif beamDirection == 0:
            if val == 1:
                Helpinggrid = placeBeam(Helpinggrid, index, 0, 1, (-1, 0))
            elif val == 2:
                Helpinggrid = placeBeam(Helpinggrid, index, 0, 0, (-1, 0))
            elif val == 4:
                Helpinggrid = placeBeam(Helpinggrid, index, 1, 1, (1, 0))
            elif val == 8:
                Helpinggrid = placeBeam(Helpinggrid, index, 1, 0, (1, 0))

    return Helpinggrid


@jit(nopython=True)
def placeBeam(Helpinggrid, index, index0, index1, direction=(0, -1)):
    """Extend the beam until it meets another wall."""

    point = (index[0] + index0, index[1] + index1)
    n = list(direction)
    while Helpinggrid[point[0]][point[1]] == 0:
        Helpinggrid[point[0]][point[1]] = 1
        point = (index[0] + index0 + n[0], index[1] + index1 + n[1])
        n[0], n[1] = (
            n[0] + direction[0],
            n[1] + direction[1],
        )

    return Helpinggrid


def runWallGrid(Grid, HelpingGrid, mutateBeams=True):
    """Run wall grid with different input depending on if beams should be added or not."""

    if mutateBeams:
        initialHelpingGrid = HelpingGrid
        initialWallGrid = createWallGrid(Grid, initialHelpingGrid)
        HelpingGrid = createAdditionalBeams(initialWallGrid, initialHelpingGrid)

    wallGrid = createWallGrid(Grid, HelpingGrid)
    return wallGrid


@jit(nopython=True)
def rectScan(Grid, wallgrid, y, x, h, w):
    """
    Scan if a there is space for a given slab at a given location
    and if it can be supported there."""

    horizontalFlag = True
    verticalFlag = True
    i = y
    if Grid.shape[1] < x + w or Grid.shape[0] < y + h:  # outside grid
        return False
    while (i - y) < h:
        j = x
        while (j - x) < w:
            if Grid[i][j] != 0:  # no space
                return False
            if i == y and wallgrid[i][j] not in [14, 12, 13]:  # first row
                horizontalFlag = False
            if i == h - 1 + y and wallgrid[i][j] not in [11, 3, 7]:  # last row
                horizontalFlag = False
            if j == x and wallgrid[i][j] not in [14, 10, 11]:
                verticalFlag = False
            if j == w - 1 + x and wallgrid[i][j] not in [13, 5, 7]:
                verticalFlag = False
            j += 1
        i += 1

    if horizontalFlag is False and verticalFlag is False:
        return False
    return True


@jit(nopython=True)
def rectPlace(Grid, wallgrid, y, x, h, w, id):
    """ If the scanning returned "True", place the slab. """

    perimeter = 0
    if rectScan(Grid, wallgrid, y, x, h, w) is True:
        perimeter = 2 * (h + w)
        i = y
        while (i - y) < h:
            j = x
            while (j - x) < w:
                Grid[i][j] = id
                j += 1
            i += 1
        id += 1
    else:
        pass
    return Grid, id, perimeter
