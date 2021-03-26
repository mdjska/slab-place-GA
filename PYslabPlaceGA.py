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


#start_time = time.time()

parser = ArgumentParser()
parser.add_argument('-i',"--numBestIndividuals", default=3, help="Number of top solutions to show")
parser.add_argument('-g',"--numGenerations", default=20, help="Number of generations")
parser.add_argument('-p',"--popSize", default=25, help="Population size")
parser.add_argument('-b',"--mutateBeams", default=False, help="Insert additional beams as part of mutation?")
parser.add_argument('-w',"--writeToFile", default=True, help="Write .json file with the solution grid")
#parser.add_argument('-s',"--slabFilePath", help="File path for slab.json")
#parser.add_argument('-t',"--gridFilePath",help="File path for grid.json")
#parser.add_argument('-y',"--sgridFilePath",help="File path for sgrid.json")
#parser.add_argument("-f", "--fingridFilePath", default =  "\\fingrid.json", help="File path for solution grid fingrid.json")

args = parser.parse_args()


def createNewChromosome(Grid, slabs, loops=1000):
    id = 2
    chromosome = []
    for i in range(0, loops):
        y = np.random.choice(np.shape(Grid)[0])
        x = np.random.choice(np.shape(Grid)[1])
        slabtype = np.random.choice(len(slabs))
        gene = [y, x, slabtype]
        chromosome.append(gene)
    return chromosome


def runChromosome(Grid, wallgrid, slabs, rotate=True, loops=1000):
    solutionGrid = []
    if rotate:
        a = []
        for i in slabs.tolist():
            a.append(i)
            a.append(i[::-1])
        slabs = a
    chromosome = createNewChromosome(Grid, slabs, loops)

    id = 2
    perimeterComb = 0
    outgrid = copy.deepcopy(Grid)
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
    emptyGridCellAfter = np.count_nonzero(solutionGrid == 0)
    score = 1 / (perimeter + (emptyGridCellAfter ** 2))
    return score


def createInitialPopulation(Grid, wallgrid, slabs, popSize, loops=1000):

    population = []
    perimeters = []
    for _ in range(popSize):
        solutionGrid, perimeterComb = runChromosome(Grid, wallgrid, slabs, loops)
        population.append(solutionGrid)
        perimeters.append(perimeterComb)

    return population, perimeters


def scorePopulation(population, perimeters):
    scores = []
    for i, solutionGrid in enumerate(population):
        scores.append(fitness(solutionGrid, perimeters[i]))
    return scores


def selectParents(population, scores):
    parentIndex = random.choices(population=range(len(population)), weights=scores, k=4)
    return parentIndex


def keepElistes(population, scores, numElistes=2):
    eliteIndex = np.argpartition(scores, -numElistes)[-numElistes:]
    elites = [population[i] for i in eliteIndex]
    return elites, eliteIndex


# in main, make population[parent] to parentA, B, C, D etc
def crossover(parentA, parentB, parentAindex, perimeters, wallgrid):
    parentBchromosome = readSolutionGrid(parentB)
    child = np.empty_like(parentA)

    try:
        id = np.amax(parentA) + 1
    except:
        print("ERROR")
        return parentA, perimeters[parentAindex]
    perimeterComb = perimeters[parentAindex]  # perimeter of parentA
    parentACopy = copy.deepcopy(parentA)
    if not parentBchromosome:
        return parentA, perimeters[parentAindex]
    for gene in parentBchromosome:
        child, id, perimeter = rectPlace(
            parentACopy,
            wallgrid,
            gene[0],
            gene[1],
            gene[2],
            gene[3],
            id,
        )
        perimeterComb += perimeter
    # plt.imshow(child, interpolation="none")
    # plt.show()
    # fig, axs = plt.subplots(ncols=3)
    # axs[0].imshow(parentA, interpolation="none")
    # axs[1].imshow(parentB, interpolation="none")
    # axs[2].imshow(child, interpolation="none")
    # plt.show()
    return child, perimeterComb


def readSolutionGrid(parent):
    chromosome = []
    i = 2
    while True:
        if i not in parent:
            break
        slabCoor = np.where(parent == i)
        h = max(slabCoor[0]) - min(slabCoor[0]) + 1
        w = max(slabCoor[1]) - min(slabCoor[1]) + 1
        y, x = (slabCoor[0][0], slabCoor[1][0])
        chromosome.append([y, x, h, w])
        i += 1
    return chromosome


def mate(population, parents, perimeters, wallgrid):
    print("Mating parents")
    children = []
    childrenPerimeter = []
    for parent in itertools.combinations(parents, 2):
        try:
            child, childPerimeter = crossover(
                population[parent[0]],
                population[parent[1]],
                parents[0],
                perimeters,
                wallgrid,
            )
            children.append(child)
            childrenPerimeter.append(childPerimeter)
        except:
            print(population)
            print(population[parent[0]], population[parent[1]])
    return children, childrenPerimeter


def mutateChildGenes(child, numGenes=2):
    childChromosome = readSolutionGrid(child)
    mutatedChildChromosome = copy.deepcopy(childChromosome)
    # number of slabs to mutate shouldn't be larger than number of slabs
    minNum = min(len(childChromosome), numGenes)
    geneToMutateIndexes = random.sample(range(len(childChromosome)), k=minNum)

    # mutate by shifting arbitrary slab in arbitrary direction
    for gene in geneToMutateIndexes:
        geneToMutate = mutatedChildChromosome[gene]
        geneSwitch = random.choice(
            [[random.choice([1, -1]), 0, 0, 0], [0, random.choice([1, -1]), 0, 0]]
        )
        geneMutated = np.add(geneToMutate, geneSwitch)
        mutatedChildChromosome[gene] = geneMutated
    return mutatedChildChromosome


def mutate(children, Grid, wallgrid, mutationProb=100, numGenes=2):
    mutatedChild = []
    perimeterComb = 0

    print("Mutating child")
    if random.randint(0, 100) < mutationProb:
        child = random.choice(children)
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
                #print(counter)
                if counter == 100:
                    print("Mutation didn't succeed")
                    # plt.imshow(wallgrid, interpolation="none")
                    # plt.show()

                    return child, perimeterComb
                restart = False

            for gene in mutatedChildChromosome:
                if (
                    rectScan(
                        tempGrid,
                        wallgrid,
                        gene[0],
                        gene[1],
                        gene[2],
                        gene[3],
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
                tempGrid, wallgrid, gene[0], gene[1], gene[2], gene[3], id
            )
            perimeterComb += perimeter
    "mutating Child done"
    return mutatedChild, perimeterComb


def createPopulation(
    elites,
    eliteIndex,
    children,
    childrenPerimeter,
    mutatedChild,
    mutatedChildPerimeter,
    perimeters,
    scores,
    currentPopulation,
    Grid,
    wallgrid,
    slabs,
    popSize,
    mutateBeams,
    loops=1500,
):
    "Creating new population!"
    newPopulationPerimeters = []
    elitesPerimeters = [perimeters[i] for i in eliteIndex]
    newPopulation = list(itertools.chain(elites, children))
    newPopulation.append(mutatedChild)

    newPopulationPerimeters.extend(elitesPerimeters + childrenPerimeter)
    newPopulationPerimeters.append(mutatedChildPerimeter)

    while len(newPopulation) < popSize:
        solutionGrid, perimeterComb = runChromosome(Grid, wallgrid, slabs, loops)
        newPopulation.append(solutionGrid)
        newPopulationPerimeters.append(perimeterComb)

    return newPopulation, newPopulationPerimeters


def main(numBestIndividuals, numGenerations, popSize, mutateBeams=False, writeToFile=False, writePath = "\\fingrid.json"):

    # Initial Run
    wallgrid = runWallGrid(grid, supportingWallGrid, mutateBeams=False)
    if mutateBeams:
        wallgrid = runWallGrid(grid, supportingWallGrid, mutateBeams=True)
    population, perimeters = createInitialPopulation(grid, wallgrid, slabs, 20)

    #plt.imshow(wallgrid, interpolation="none")
    #plt.show()
    #plt.imshow(grid, interpolation="none")
    #plt.show()
    # GENERATIONS
    for i in range(numGenerations):

        print("GENERATION: ", i)
        try:
            scores = scorePopulation(population, perimeters)
        except:
            print("Empty grid")

        parents = selectParents(population, scores)

        elites, eliteIndex = keepElistes(population, scores)

        children, childrenPerimeter = mate(population, parents, perimeters, wallgrid)

        mutatedChild, mutatedChildPerimeter = mutate(
            children, grid, wallgrid, mutationProb=100
        )

        newPopulation, newPerimeters = createPopulation(
            elites,
            eliteIndex,
            children,
            childrenPerimeter,
            mutatedChild,
            mutatedChildPerimeter,
            perimeters,
            scores,
            population,
            grid,
            wallgrid,
            slabs,
            popSize,
            mutateBeams=True,
            loops=2000,
        )

        population = copy.deepcopy(newPopulation)
        perimeters = copy.deepcopy(newPerimeters)
        wallgrid = copy.deepcopy(wallgrid)

    print("#########THE BEST INDIVIADUALS")

    ### PLOT
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
        #axs[0].imshow(wallgrid, interpolation="none")
        #axs[1].imshow(population[best], interpolation="none")
        #plt.show()
    
    # WRITE TO FILE
    print(bestIndividuals[0])
    if writeToFile:

        path_ = pathlib.Path(__file__).parent.absolute()
        writePath = str(path_) + '\\fingrid.json'
        print("path: ", writePath)
        with open(writePath,'w') as myfile:
            json.dump(bestIndividuals[0].tolist(),myfile)
        print("File written")
        #if writeToFile:

    return "Fin"


#main(numBestIndividuals=3, numGenerations=20, popSize=20, mutateBeams= False, writeToFile =True, writePath = r"C:\Users\marti\Documents\CN3 specialprojekt\RevitTests\TestfingridRevit.json")


if __name__ == "__main__":
    path_ = pathlib.Path(__file__).parent.absolute()
    writePath = str(path_) + '\\fingrid.json'
    slabs = load_json(str(path_) + '\\slabsRevit.json')
    grid = load_json(str(path_) + '\\gridRevit.json')
    sgrid = load_json(str(path_) + '\\sgridRevit.json')
    

    emptygrid = grid.copy()
    sgrid[sgrid == 0] = -1
    sgrid[sgrid == 2] = 0
    supportingWallGrid = np.flip(sgrid, axis = 0)
    grid = np.flip(grid, axis = 0)

    #try:
        #main(numBestIndividuals=int(args.numBestIndividuals), numGenerations=int(args.numGenerations), popSize=int(args.popSize), mutateBeams=bool(args.mutateBeams), writeToFile = bool(args.writeToFile), writePath = str(args.fingridFilePath))
    main(numBestIndividuals=int(args.numBestIndividuals), numGenerations=int(args.numGenerations), popSize=int((args.popSize)), mutateBeams=bool(args.mutateBeams), writeToFile = bool(args.writeToFile))
     
    #except:
    #    print("Something went wrong")
    

#print("--- GA: %s seconds ---" % (time.time() - start_time))
