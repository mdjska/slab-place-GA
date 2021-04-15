from WallsAndBeams import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
import copy
from statistics import median
import itertools
from argparse import ArgumentParser
import os
import sys
import pathlib
import cProfile, pstats, io


def profile(fnc):

    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):

        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


start_time = time.time()

parser = ArgumentParser()
parser.add_argument(
    "-i", "--numBestIndividuals", default=3, help="Number of top solutions to show"
)
parser.add_argument("-g", "--numGenerations", default=20, help="Number of generations")
parser.add_argument("-p", "--popSize", default=25, help="Population size")
parser.add_argument("-l", "--loops", default=2000, help="Number of chromosome loops")
parser.add_argument("-m", "--mutationProb", default=10, help="mutation probability")
parser.add_argument(
    "-b",
    "--mutateBeams",
    default=False,
    help="Insert additional beams as part of mutation?",
)
parser.add_argument(
    "-w", "--writeToFile", default=True, help="Write .json file with the solution grid"
)

args = parser.parse_args()


@jit(nopython=True)
def createNewChromosome(Grid, slabs, loops=1000):
    """Create number of random placements of random slabs to try.
    Collection of placement and slab is called a chromosome."""

    id = 2
    chromosome = []
    for i in range(0, loops):
        y = np.random.choice(np.shape(Grid)[0])
        x = np.random.choice(np.shape(Grid)[1])
        slabtype = np.random.choice(len(slabs))
        gene = [y, x, slabtype]
        chromosome.append(gene)
    return chromosome


def runChromosome(Grid, wallgrid, slabs, loops=1000):
    """ Try placing the chromosome genes by running rectPlace. """

    solutionGrid = []

    chromosome = createNewChromosome(Grid, slabs, loops)

    id = 2
    perimeterComb = 0
    outgrid = Grid.copy()
    for gene in chromosome:
        solutionGrid, id, perimeter = rectPlace(
            outgrid,
            wallgrid,
            gene[0],
            gene[1],
            int(slabs[gene[2]][0]),
            int(slabs[gene[2]][1]),
            id,
        )
        perimeterComb += perimeter
    return solutionGrid, perimeterComb


def fitness(solutionGrid, perimeter):
    """Calculate the fitness score for each individual."""

    emptyGridCellAfter = np.count_nonzero(solutionGrid == 0)
    score = 1 / (perimeter + (emptyGridCellAfter ** 2))
    return score


def createInitialPopulation(Grid, wallgrid, slabs, popSize, loops=1000):
    """Create initial population of solutions. """

    population = []
    perimeters = []
    for _ in range(popSize):
        solutionGrid, perimeterComb = runChromosome(Grid, wallgrid, slabs, loops)
        population.append(solutionGrid)
        perimeters.append(perimeterComb)

    return population, perimeters


def scorePopulation(population, perimeters):
    """Get the scores of all solutions (individuals) in a population. """

    scores = []
    for i, solutionGrid in enumerate(population):
        scores.append(fitness(solutionGrid, perimeters[i]))
    return scores


def selectParents(population, scores):
    """Choose 4 parents from the population with probability equal to it's fitness score. """

    parentIndex = random.choices(population=range(len(population)), weights=scores, k=4)
    return parentIndex


def keepElistes(population, scores, numElistes=2):
    """Keep the two best solutions to next generation without change. """

    eliteIndex = np.argpartition(scores, -numElistes)[-numElistes:]
    elites = [population[i] for i in eliteIndex]
    return elites, eliteIndex


# in main, make population[parent] to parentA, B, C, D etc
def crossover(parentA, parentB, parentAindex, perimeters, wallgrid):
    """Mate two parents by trying to combine them, and chosing parentA
    if there is overlap."""

    parentBchromosome = readSolutionGrid(parentB)
    child = np.empty_like(parentA)
    try:
        id = np.amax(parentA) + 1
    except Exception:
        print(Exception)
        input()
        return parentA, perimeters[parentAindex]
    perimeterComb = perimeters[parentAindex]  # perimeter of parentA
    parentACopy = parentA.copy()
    if parentBchromosome.size == 0:
        return parentA, perimeters[parentAindex]
    for gene in parentBchromosome:
        child, id, perimeter = rectPlace(
            parentACopy,
            wallgrid,
            int(gene[0]),
            int(gene[1]),
            int(gene[2]),
            int(gene[3]),
            id,
        )
        perimeterComb += perimeter

    return child, perimeterComb


# @jit(nopython = True)
def readSolutionGrid(parent):
    """Return a chromosome given a solution grid. """

    chromosome = np.empty((0, 4))
    i = 2
    while True:
        if i not in parent:
            break
        slabCoor = np.where(parent == i)
        h = max(slabCoor[0]) - min(slabCoor[0]) + 1
        w = max(slabCoor[1]) - min(slabCoor[1]) + 1
        y, x = (slabCoor[0][0], slabCoor[1][0])
        params = np.array([[y, x, h, w]])
        chromosome = np.append(chromosome, params, axis=0)
        i += 1
    return chromosome


def mate(population, parents, perimeters, Grid, wallgrid, mutationProb):
    """Combine parents with eachother and mate them, adding a possibility to mutate.
    Return children solutions."""

    children = []
    childrenPerimeter = []
    for parent in itertools.combinations(parents, 2):
        childTemp, childPerimeterTemp = crossover(
            population[parent[0]],
            population[parent[1]],
            parents[0],
            perimeters,
            wallgrid,
        )
        child, childPerimeter = mutate(
            childTemp, childPerimeterTemp, Grid, wallgrid, mutationProb, numGenes=2
        )
        children.append(child)
        childrenPerimeter.append(childPerimeter)

    return children, childrenPerimeter


def mutateChildGenes(child, numGenes=2):
    """Mutate by shifting arbitrary slab in arbitrary direction. """

    childChromosome = readSolutionGrid(child)
    mutatedChildChromosome = childChromosome.copy()
    # number of slabs to mutate shouldn't be larger than number of slabs
    minNum = min(len(childChromosome), numGenes)
    geneToMutateIndexes = random.sample(range(len(childChromosome)), k=minNum)

    for gene in geneToMutateIndexes:
        geneToMutate = mutatedChildChromosome[gene]
        geneSwitch = random.choice(
            [[random.choice([1, -1]), 0, 0, 0], [0, random.choice([1, -1]), 0, 0]]
        )
        geneMutated = np.add(geneToMutate, geneSwitch)
        mutatedChildChromosome[gene] = geneMutated
    return mutatedChildChromosome


def mutate(child, childPerimeter, Grid, wallgrid, mutationProb, numGenes=2):
    """With a mutation probability, try to mutate a solution, scan to see
    if the solution is valid and then create it by placing the slab."""

    mutatedChild = []
    perimeterComb = 0
    if random.randint(0, 100) < mutationProb:
        # Initial Mutation
        mutatedChildChromosome = mutateChildGenes(child, numGenes)
        tempGrid = copy.deepcopy(Grid)

        # Check if mutation is valid, if not, mutate again
        i = 0
        counter = 0
        restart = False
        while i < len(mutatedChildChromosome):
            if restart:
                mutatedChildChromosome = mutateChildGenes(child, numGenes)
                counter += 1
                if counter == 100:
                    print("Mutation didn't succeed")

                    return child, childPerimeter
                restart = False

            for gene in mutatedChildChromosome:
                if (
                    rectScan(
                        tempGrid,
                        wallgrid,
                        int(gene[0]),
                        int(gene[1]),
                        int(gene[2]),
                        int(gene[3]),
                    )
                    is True
                ):
                    i += 1
                else:
                    i = 0
                    restart = True
                    break

        # Generate solution from mutated chromosomes
        id = 2

        for gene in mutatedChildChromosome:
            mutatedChild, id, perimeter = rectPlace(
                tempGrid,
                wallgrid,
                int(gene[0]),
                int(gene[1]),
                int(gene[2]),
                int(gene[3]),
                id,
            )
            perimeterComb += perimeter
    else:
        return child, childPerimeter
    return mutatedChild, perimeterComb


def createPopulation(
    elites,
    eliteIndex,
    children,
    childrenPerimeter,
    perimeters,
    Grid,
    wallgrid,
    slabs,
    popSize,
    loops,
):
    """Create a new population. Fill the spots remaining after children and elites,
    by random solutions."""

    newPopulationPerimeters = []
    elitesPerimeters = [perimeters[i] for i in eliteIndex]
    newPopulation = list(itertools.chain(elites, children))

    newPopulationPerimeters.extend(elitesPerimeters + childrenPerimeter)

    while len(newPopulation) < popSize:
        solutionGrid, perimeterComb = runChromosome(Grid, wallgrid, slabs, loops)
        newPopulation.append(solutionGrid)
        newPopulationPerimeters.append(perimeterComb)

    return newPopulation, newPopulationPerimeters


# @profile
def main(
    numBestIndividuals,
    numGenerations,
    popSize,
    loops,
    mutationProb,
    mutateBeams=False,
    writeToFile=False,
    writePath="\\fingrid.json",
):
    """Main run mechanism. Initiate first population, loop over generations,
    plot the best individuals and write the result to a .json file."""

    # Initial Run
    wallgrid = runWallGrid(grid, supportingWallGrid, mutateBeams=False)
    if mutateBeams:
        wallgrid = runWallGrid(grid, supportingWallGrid, mutateBeams=True)

    rotate = True
    if rotate:
        shape = np.shape(slabsSingle)[0] * 2
        slabs = np.empty((shape, 2))
        slabs[::2] = slabsSingle
        slabs[1::2] = np.fliplr(slabsSingle)

    population, perimeters = createInitialPopulation(grid, wallgrid, slabs, 20)

    # GENERATIONS
    for i in range(numGenerations):
        print("GENERATION: ", i)
        try:
            scores = scorePopulation(population, perimeters)
        except:
            print("Empty grid")

        parents = selectParents(population, scores)

        elites, eliteIndex = keepElistes(population, scores)

        children, childrenPerimeter = mate(
            population, parents, perimeters, grid, wallgrid, mutationProb
        )

        newPopulation, newPerimeters = createPopulation(
            elites,
            eliteIndex,
            children,
            childrenPerimeter,
            perimeters,
            grid,
            wallgrid,
            slabs,
            popSize,
            loops,
        )

        population = copy.deepcopy(newPopulation)
        perimeters = copy.deepcopy(newPerimeters)
        wallgrid = copy.deepcopy(wallgrid)

    print("#########THE BEST INDIVIADUALS")

    ### PLOT - ONLY IF RUNNING OUTSIDE GRASSHOPPER
    bestIndex = np.argpartition(scores, -numBestIndividuals)[-numBestIndividuals:]
    bestIndividuals = [population[i] for i in bestIndex]
    for best in bestIndex:
        print("Best Individual scores: ", scores[best])
        colors = [(1, 1, 1)] + [
            (random.randint(0, 1), random.randint(0, 1), random.randint(0, 1))
            for i in range(255)
        ]
        new_map = matplotlib.colors.LinearSegmentedColormap.from_list(
            "new_map", colors, N=256
        )
        fig, axs = plt.subplots(ncols=2)
        # axs[0].imshow(wallgrid, interpolation="none")
        # axs[1].imshow(population[best], interpolation="none")
        # plt.show()

    # WRITE BEST RESULT TO FILE
    print(bestIndividuals[0])
    if writeToFile:

        path_ = pathlib.Path(__file__).parent.absolute()
        writePath = str(path_) + "\\fingrid.json"
        print("path: ", writePath)
        with open(writePath, "w") as myfile:
            json.dump(bestIndividuals[0].tolist(), myfile)
        print("File written")

    return "Fin"


# IF RUNNING OUTSIDE GRASSHOPPER
# main(numBestIndividuals=3, numGenerations=20, popSize=20, mutateBeams= False, writeToFile =True, writePath = r"C:\Users\marti\Documents\CN3 specialprojekt\RevitTests\TestfingridRevit.json")


if __name__ == "__main__":
    """Argument parser solution for running from Grasshopper (through command-line tool) """

    path_ = pathlib.Path(__file__).parent.absolute()
    writePath = str(path_) + "\\fingrid.json"
    slabsSingle = load_json(str(path_) + "\\slabsRevit.json")
    grid = load_json(str(path_) + "\\gridRevit.json")
    sgrid = load_json(str(path_) + "\\sgridRevit.json")

    emptygrid = grid.copy()
    sgrid[sgrid == 0] = -1
    sgrid[sgrid == 2] = 0
    supportingWallGrid = np.flip(sgrid, axis=0)
    grid = np.flip(grid, axis=0)

    try:
        # main(numBestIndividuals=int(args.numBestIndividuals), numGenerations=int(args.numGenerations), popSize=int(args.popSize), mutateBeams=bool(args.mutateBeams), writeToFile = bool(args.writeToFile), writePath = str(args.fingridFilePath))
        main(
            numBestIndividuals=int(args.numBestIndividuals),
            numGenerations=int(args.numGenerations),
            popSize=int((args.popSize)),
            loops=int((args.loops)),
            mutationProb=int((args.mutationProb)),
            mutateBeams=bool(int(args.mutateBeams)),
            writeToFile=bool(int(args.writeToFile)),
        )

    except Exception as e:
        print(e)
        input()


print("--- GA: %s seconds ---" % (time.time() - start_time))
