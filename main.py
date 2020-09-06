import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import random
import csv
from datetime import datetime
from matplotlib.lines import Line2D
import time
import sys
result = nx.DiGraph()
g = nx.DiGraph()
data = pd.read_csv('network_combined.csv', error_bad_lines=False)


#--------------Build Graph--------------
class Graph :

    def build_graph_and_get_positions(data , g):
        ''' Builds graph and returns a dictionary where the key is the node and value is lat/lon '''

        # Extracting data
        distances = data.loc[:, 'd'].values.tolist()
        stops = data.loc[:, 'from_stop_I':'to_stop_I'].values.tolist()
        durations = data.loc[:, 'duration_avg'].values.tolist()
        route_type_list = data.loc[:, 'route_type'].values.tolist()
        route_imp_list = data.loc[:, 'route_importance'].values.tolist()
        price = data.loc[:, 'Price'].values.tolist()
        stars = data.loc[:, 'star_reviews'].values.tolist()
        walking = data.loc[:, 'iswalk'].values.tolist()
        distance_walk = data.loc[:, 'd_walk'].values.tolist()
        rush_hours = data.loc[:, 'rush_hour'].values.tolist()

        route_dic = {0: 'Tram', 1: 'Subway', 2: 'Rail', 3: 'Bus', 4: 'Ferry', 5: 'Cable car', 6: 'Gondola',
                     7: 'Funicular', 8: 'Private car', 9: 'bicycle'}

        # Tram: green   Subway: orange   Rail: pink   Bus: black   Ferry: blue   Cable car: yellow   Gondola: purple   Funicular: grey   Walking: red
        route_colors = {'Tram': 'g', 'Subway': 'orange', 'Rail': 'm', 'Bus': 'k', 'Ferry': 'powderblue',
                        'Cable car': 'y',
                        'Gondola': 'c', 'Funicular': '0.75', 'Walking': 'r', 'Private car': 'b',
                        'bicycle': 'hotpink'}

        # Building graph
        for num in range(len(stops)):
            g.add_edge(stops[num][0], stops[num][1], length=distances[num] / 10, duration=durations[num],
                       route_type=route_dic[route_type_list[num]], route_importance=route_imp_list[num],
                       price=price[num],
                       stars=stars[num], distance_walk=distance_walk[num], walking=walking[num],
                       rush_hours=rush_hours[num],
                       color=route_colors[route_dic[route_type_list[num]]])

        # Reading file and extracting data to get latitude and longitude
        info = pd.read_csv('network_nodes.csv')
        positions = info.loc[:, 'lat':'lon'].values.tolist()
        stops = info.loc[:, 'stop_I'].values.tolist()

        pos = {}
        for num in range(len(stops)):
            pos[stops[num]] = positions[num]

        return pos


    # Returns edges with attributes
    def get_edges_with_attributes(g):
        ''' Returns and prints edges with attributes '''
        edges_with_att = {}

        edges = g.edges()
        for obj1, obj2 in edges:
            edges_with_att[f'{obj1}:{obj2}'] = g.get_edge_data(obj1, obj2)
            # print(f'{obj1}:{obj2} {g.get_edge_data(obj1, obj2)}')

        return edges_with_att


    def get_nodes(g=g):
        ''' Returns list of nodes. '''

        return g.nodes()


    # Draws and showcases graph
    def draw_graph(data, g, isResult , price , star , duration):
        pos = Graph.build_graph_and_get_positions(data, g)

        colors = nx.get_edge_attributes(g, 'color').values()
        node_colors = ['red']
        for num in range(1, len(g.nodes) - 1):
            node_colors.append('#1f78b4')
        node_colors.append('green')
        if isResult == True:
            nx.draw_networkx(g, with_labels=1, node_size=50, font_size=5, arrows=1, arrow_size=20, pos=pos,
                             edge_color=colors, node_color=node_colors)
        else:
            nx.draw_networkx(g, with_labels=1, node_size=50, font_size=5, arrows=1, arrow_size=20, pos=pos,
                             edge_color=colors)
        plt.axis('off')
        if isResult == False:
            plt.title('Winnipeg graph', loc='left', fontsize=7)
        else:
            plt.title("Price " + str(price) + " SR" + "\nScenery " + str(
                star) + " / 5" + "\nDuration " + str(duration) + " Min",
                      loc='left', fontsize=9)

        fig = plt.gcf()
        fig.canvas.set_window_title('Original graph')

        # Creating color map

        color_palette = ['g', 'orange', 'm', 'k', 'b', 'y', 'c', '0.75', 'r', 'powderblue', 'hotpink']
        labels = ['Tram', 'Subway', 'Rail', 'Bus', 'Ferry', 'Cable car', 'Gondola', 'Funicular', 'Walking',
                  'Private car', 'bicycle']
        lines = []
        for i in range(11):
            lines.append(mpatches.Patch(color=color_palette[i], label=labels[i]))
        if isResult == True:
            lines.append(
                Line2D([0], [0], marker='o', color='w', label='Source node', markerfacecolor='r', markersize=12))
            lines.append(
                Line2D([0], [0], marker='o', color='w', label='Goal node', markerfacecolor='green', markersize=12))

        plt.legend(handles=lines, loc='lower left', ncol=2, fontsize='small')

        plt.show()

    # Show result
    def ShowResult(finalSol):
          with open('Result.csv', 'w', newline='') as file:
               writer = csv.writer(file)
               writer.writerow(
                ["from_stop_I", "to_stop_I", "d", "duration_avg", "n_vehicles", "route_type", "d_walk", "iswalk",
                 "route_importance", "Price", "rush_hour", "star_reviews", "Duration_Walk_min"])

               solution = finalSol[0].getSolution()
               # Read data from CSV file
               for j in range(len(solution) - 1):
                    info = ((data[(data['from_stop_I'] == solution[j]) & (data['to_stop_I'] == solution[j + 1])]))
                    if ((j == finalSol[0].getmutatedInd()) and (finalSol[0].gethasmutation() == True)):
                        dat = [info.iloc[1]['from_stop_I'], info.iloc[1]['to_stop_I'], info.iloc[1]['d'],
                               info.iloc[1]['duration_avg'],
                               info.iloc[1]['n_vehicles'], info.iloc[1]['route_type'], info.iloc[1]['d_walk'],
                               info.iloc[1]['iswalk'],
                               info.iloc[1]['route_importance'],
                               info.iloc[1]['Price'],
                               info.iloc[1]['rush_hour'], info.iloc[1]['star_reviews'],
                               info.iloc[1]['Duration_Walk_min']]
                    else:

                        dat = [info.iloc[0]['from_stop_I'], info.iloc[0]['to_stop_I'], info.iloc[0]['d'],
                               info.iloc[0]['duration_avg'],
                               info.iloc[0]['n_vehicles'], info.iloc[0]['route_type'], info.iloc[0]['d_walk'],
                               info.iloc[0]['iswalk'],
                               info.iloc[0]['route_importance'],
                               info.iloc[0]['Price'],
                               info.iloc[0]['rush_hour'], info.iloc[0]['star_reviews'],
                               info.iloc[0]['Duration_Walk_min']]

                    writer.writerow(dat)

          da = pd.read_csv('Result.csv', error_bad_lines=False)
          Graph.build_graph_and_get_positions(da, result)
          Graph.draw_graph(da, result, True, round(finalSol[0].getPrice(), 2),
                              round(finalSol[0].getStars(), 1), round(finalSol[0].getDuration()))


#------------------------Start Genetic Algorithm------------------------

class Chromosome:

    def __init__(self, solution, duration ,price , stars, fitness, hasmutation, mutatedInd):
        self.solution = solution
        self.duration = duration
        self.price = price
        self.stars = stars
        self.fitness = fitness
        self.hasmutation =hasmutation
        self.mutatedInd =mutatedInd

    def getDuration(self):
        return self.duration

    def getSolution(self):
        return self.solution

    def getStars(self):
        return self.stars

    def getPrice(self):
        return self.price

    def setDuration(self,duration):
        self.duration = duration

    def setPrice(self,price):
        self.price = price

    def setStars(self,stars):
        self.stars = stars

    def setSolution(self,solution):
        self.solution = solution

    def setFitness(self, fitness):
        self.fitness = fitness

    def getFitness(self):
        return self.fitness

    def sethasmutation(self,hasmutation):
        self.hasmutation= hasmutation

    def gethasmutation(self):
        return self.hasmutation

    def setmutatedInd(self,mutatedInd):
        self.mutatedInd= mutatedInd

    def getmutatedInd(self):
        return self.mutatedInd


class GA:
    # Generate the Initial Solutions by creating a list of chromosomes(possible paths)
    def generatepop(solutions):

        population = []
        for solution in solutions:
            chrom = Chromosome(solution, 0, 0, 0, 0, False,0)
            population.append(chrom)
        return population

    # Evaluate solutions(chromosomes) using Weighted Average Ranking (WAR)
    def calcFitness(population, preferences):
        data = pd.read_csv('network_combined.csv')
        for i in population:
            solution = i.getSolution()
            duration = 0
            stars = 0
            price = 0
            if i.gethasmutation() == False:
            # Read data from CSV file
                for j in range(len(solution) - 1):
                    info = (data[(data['from_stop_I'] == solution[j]) & (data['to_stop_I'] == solution[j + 1])])
                    now = datetime.now()
                    currentTime = now.strftime("%H")

                    if "No rush hours" != (
                    info.iloc[0]['rush_hour']):  # Check if it is a rush hour to calculate duration it takes
                        rushHours = info.iloc[0]['rush_hour'].split(',')
                        for rushHour in rushHours:
                            m = rushHour.index(':')
                            if (int(currentTime) == int(rushHour[:m])):
                                duration += (info.iloc[0]['duration_avg']) / 2
                    duration += info.iloc[0]['duration_avg']
                    price += int(info.iloc[0]['Price'])
                    stars += int(info.iloc[0]['star_reviews'])
            else:  # If it has mutated, will get the data of preferences that set in mutation function
                stars = i.getStars()
                price = i.getPrice()
                duration = i.getDuration()
            set_duration = False
            set_stars = False
            set_price = False
            if 1 in preferences:
                set_duration = True
            i.setDuration(duration)
            if 2 in preferences:
                set_stars = True
            i.setStars(stars / len(solution))  # to normalize the scenery to be in the range 0 - 5
            if 3 in preferences:
                set_price = True
            i.setPrice(price)
        #sort the preferences in order to make the ranking
        starRank = GA.sortStars(population)
        piceRank = GA.sortPrice(population)
        durationRank = GA.sortDuration(population)

        for i in range(len(population)):
            fiteness = 0
            if set_duration:
                fiteness += durationRank[i]
            if set_stars:
                fiteness += starRank[i]
            if set_price:
                fiteness += piceRank[i]
            population[i].setFitness((fiteness / 3))
        return population

    # Rank preference (Duration); the path that has the highest duration will set to 1
    def sortDuration(population):
        duration = []
        sortduration = []
        for i in range(len(population)):
            duration.insert(i, population[i].getDuration())
        sortduration = duration[:]
        prev = -1
        rank = 0
        for i in range(len(population)):
            curr = max(duration)
            index = duration.index(max(duration))
            duration[index] = -1
            if prev == curr:
                sortduration[index] = rank
            else:
                rank = rank + 1
                sortduration[index] = rank
            prev = curr
        return sortduration

    # Rank preference (Scenery); the path that has the lowest stars will set to 1
    def sortStars(population):
        star = []
        sortStar = []
        for i in range(len(population)):
            star.insert(i, population[i].getStars())
        sortStar = star[:]
        prev = -1
        rank = 0
        maxstar = max(star) + 1
        for i in range(len(population)):
            curr = min(star)
            index = star.index(min(star))
            star[index] = maxstar
            if prev == curr:
                sortStar[index] = rank
            else:
                rank = rank + 1
                sortStar[index] = rank
            prev = curr
        return sortStar

    # Rank preference (Price); the path that has the highest price will set to 1
    def sortPrice(population):
        price = []
        sortprice = []
        for i in range(len(population)):
            price.insert(i, population[i].getPrice())
        sortprice = price[:]
        prev = -1
        rank = 0
        for i in range(len(population)):
            curr = max(price)
            index = price.index(max(price))
            price[index] = -1
            if prev == curr:
                sortprice[index] = rank
            else:
                rank = rank + 1
                sortprice[index] = rank
            prev = curr
        return sortprice

    # Parents Selection using Roulette Wheel algorithm
    def selection(population):
        check = False
        # Generating PDF
        parent = []
        pdf = []
        total_sum = 0
        for i in range(len(population)):
            total_sum += population[i].getFitness()

        for i in range(len(population)):
            pdf.append(population[i].getFitness() / total_sum)

        current = 0
        i = 0
        parent1_index = -1
        parent2_index = -1
        commonIndex1 = 0
        commonIndex2 = 0

        entered = False
        pick = random.uniform(0, total_sum)
        while check == False:
            current += pdf[i]
            if current > pick:
                child = population[i]
                parent.append(copy.copy(child))
                current = 0
                pick = random.uniform(0, total_sum)
            if len(parent) == 1 and entered == False:
                parent1_index = i
                entered = True
            if len(parent) == 2:
                parent2_index = i
                parent1 = parent[0].getSolution()
                parent2 = parent[1].getSolution()
                y = set(parent1) & set(parent2)  # Check if it selects the same chromosome
                if len(y) == (len(parent1) or len(parent2)):
                    parent.pop(0)
                    parent.pop(0)
                    entered = False
                    i = -1
                    continue
                for ii in range(1, len(
                        parent1) - 1):  # Check if there is a common gene (Station) between the selected chromosomes
                    for iii in range(1, len(parent2) - 1):
                        if parent1[ii] == parent2[iii]:
                            commonIndex2 = iii
                            commonIndex1 = ii
                            check = True
                            break
                    if check == True:
                        break

                if check != True:
                    parent.pop(0)
                    parent.pop(0)
                    entered = False
                    i = -1

            if i == len(pdf) - 1:
                i = 0
            else:
                i = i + 1
        sol = [0]
        population[parent1_index].setSolution(sol)
        population[parent2_index].setSolution(sol)

        return parent, commonIndex1, commonIndex2, parent1_index, parent2_index

    # Single-point crossover type, by swapping all data beyond the common gene (station) in selected parent
    def crossover(parent, commonIndex1, commonIndex2):

        offspring = []

        offspring.append(copy.deepcopy(parent[0]))
        offspring.append(copy.deepcopy(parent[1]))

        x = offspring[0].getSolution()
        y = offspring[1].getSolution()

        m = x[commonIndex1:]
        n = y[commonIndex2:]

        for i in range(len(x), commonIndex1, -1):
            x.pop(i - 1)

        for i in range(len(y), commonIndex2, -1):
            y.pop(i - 1)

        x = x + n
        y = y + m

        offspring[0].setSolution(x)
        offspring[1].setSolution(y)

        return offspring

    # mutate the offsprings by changing the edge attributes between two genes (stations)
    def mutation(children):
        probability = 0.015

        duration = 0
        price = 0
        stars = 0
        d = 0
        while d < 2:

            pick = random.uniform(0, 1)

            if pick <= probability:
                childrensol = children[d].getSolution()

                for i in range(len(childrensol) - 1):
                    numOfEdges = len(
                        data.loc[(data['from_stop_I'] == childrensol[i]) & (data['to_stop_I'] == childrensol[i + 1])])
                    info = data.loc[(data['from_stop_I'] == childrensol[i]) & (data['to_stop_I'] == childrensol[i + 1])]
                    if numOfEdges > 1:
                        children[d].setmutatedInd(i)
                        children[d].sethasmutation(True)
                        duration = children[d].getDuration()
                        duration = duration - (info.iloc[0]['duration_avg'])
                        duration += info.iloc[1]['duration_avg']
                        children[d].setDuration(duration)

                        price = children[d].getPrice()
                        price = price - (info.iloc[0]['Price'])
                        price += info.iloc[1]['Price']
                        children[d].setPrice(price)

                        stars = children[d].getStars()
                        stars = stars - (info.iloc[0]['star_reviews'])
                        stars += info.iloc[1]['star_reviews']
                        children[d].setStars(stars / len(childrensol))
                        break

            d = d + 1
        return children

    # Select two chromosomes that have the most higher fitness from parent and offsprings
    def BestSol(parent, offspring, preferences):
        fitness = []
        best = []
        pop = []
        pop.append(parent[0])
        pop.append(parent[1])
        pop.append(offspring[0])
        pop.append(offspring[1])
        GA.calcFitness(pop, preferences)

        for i in pop:
            fitness.append(i.getFitness())

        deleted = -1

        for i in range(len(fitness)):
            index = fitness.index(max(fitness))
            best.append(pop[index])
            fitness[index] = deleted
            if len(best) == 2:
                if (best[0].getSolution() == best[1].getSolution()):
                    best.pop(0)
                    best.pop(0)
                    if pop[0].getFitness() <= pop[3].getFitness():
                        best.append(copy.deepcopy(pop[3]))
                    else:
                        best.append(copy.deepcopy(pop[0]))
                    if pop[1].getFitness() <= pop[2].getFitness():
                        best.append(copy.deepcopy(pop[2]))
                    else:
                        best.append(copy.deepcopy(pop[1]))
                break

        return best

    # Select the best 4 final solutions
    def finalSol(population):
        fitness = []
        best = []
        for i in population:
            fitness.append(i.getFitness())

        for i in range(len(fitness)):
            index = fitness.index(max(fitness))
            best.append(population[index])
            fitness[index] = -1
            if len(best) == 4:
                break

        return best


    def updt(total, progress):
        """
        Displays or updates a console progress bar.

        Original source: https://stackoverflow.com/a/15860757/1391441
        """
        barLength, status = 20, ""
        progress = float(progress) / float(total)
        if progress >= 1.:
            progress, status = 1, "\r\n"
        block = int(round(barLength * progress))
        text = "\r[{}] {:.0f}% {}".format(
            "#" * block + "-" * (barLength - block), round(progress * 100, 0),
            status)
        sys.stdout.write(text)
        sys.stdout.flush()

#---------------------Main---------------------
if __name__ == "__main__":

    # Build Graph
    Graph.build_graph_and_get_positions(data, g)
    edges_with_att = Graph.get_edges_with_attributes(g)
    # Show graph (uncomment the below line in order to display the original graph
    Graph.draw_graph(data ,g, False,0,0,0)

    print("Please enter the number of preference:\n", "1: less duration\n", "2: Highest rating\n", "3: less price\n 0: to exit")

    preferences = []
    # Read the required preferences from the user
    input2= 9
    while input2!="0":
        input2 = input()
        preferences.append(int(input2))

    solutions =[]
    # Generate path by depth-first search algorithm
    paths = nx.all_simple_paths(g, source=0, target=18)

    pathss = []
    for path in paths:
        pathss.append(path)

    pathsize = len(pathss)
    for path in range(pathsize):
        random_path = random.choice(pathss)
        solutions.append(random_path)
        pathss.remove(random_path)
        if len(solutions) == 200:
            break

    population= GA.generatepop(solutions) # Generate initial population population
    GA.calcFitness(population, preferences) # Calculate fitness function
    no_iteration=300
    for generation in range(no_iteration):

        time.sleep(.1)
        GA.updt(no_iteration, generation + 1)

        # Parents selection
        parents, commonInd1, commonInd2, ind1, ind2 = GA.selection(population)
        
        # Single-point crossover type
        child = GA.crossover(parents, commonInd1, commonInd2)
        
        # mutation
        child2 = GA.mutation(child)
        
        # Select two chromosomes that have the most higher fitness from parent and offsprings
        bestsol = GA.BestSol(parents, child2,preferences)

        population[ind2] = bestsol[0]
        population[ind1] = bestsol[1]


        GA.calcFitness(population, preferences)
    finalSolution = GA.finalSol(population)
    Graph.ShowResult(finalSolution)




