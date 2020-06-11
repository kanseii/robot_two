import numpy as np
from math import *
import pygame
import random
import pickle
import matplotlib.pyplot as plt
vec = pygame.math.Vector2

#TODO
# 初始化权重保存


# ANN 结构
input_size = 4
hidden_size = 100
output_size = 2
std = 1e-1

# Robot config
POS_LIGHT = np.array([0,0])
ANGLE = 0
D = 1
VMAX = 10



# GA params
pop_size = 100
gene_size = 142       # DNA length
K_tournament = 10


def load(filename):
    try:
        with open(filename + ".pkl", 'rb') as file:
            ga_in = pickle.load(file)
    except (FileNotFoundError):
        print("Error loading the file. Please check if the file exists.")
    return ga_in

def relu(x):    
    return np.maximum(0, x)


    
def forward_propagate(X, theta1, theta2):
        m = X.shape[0]
        a1 = np.insert(X, 0, values=np.zeros(m), axis=1)
        z2 = a1 * theta1.T
        a2 = np.insert(relu(z2), 0, values=np.zeros(m), axis=1)
        z3 = np.array(a2 * theta2.T)
        z3 -= z3.max(axis=-1, keepdims=True)  # 防止溢出
        exp_z3 = np.array(np.exp(z3))
        h = exp_z3 / np.sum(exp_z3, axis=1, keepdims=True)
        return h*VMAX

class Light:
    def __init__(self,pos,id):
        self.pos = pos
        self.id = id

class Robot(object):
    x = random.randrange(30, 800 - 30)
    y = random.randrange(30, 600 - 30)
    _pos = np.array([x,y])
    # 机器人属性
    def __init__(self, id = None,pos=_pos, velocity=[0, -VMAX]):
        self.angle = 0      # 初始化角度 竖直向上为0度
        self.pos = np.array(pos)
        self.velocity = vec(velocity)
        self.vl = 0         # 初始化左轮速度
        self.vr = 0         # 初始化右轮速度
        self.id = id
        self.imageSrc = pygame.image.load("50.png")
        self.rect = self.imageSrc.get_rect()
        self.rect = self.rect.move(pos[0], pos[1])  # 初始化位置

    def direction(self):
        vel = np.linalg.norm(self.velocity)
        return self.velocity/vel 

    def get_angles(self,light):
        self.pos = self.rect.center
        x1 = self.pos[0] + D*np.cos((ANGLE + 180 - self.angle)*np.pi/180)/2
        y1 = self.pos[1] + D*np.sin((ANGLE + 180 - self.angle)*np.pi/180)/2
        x2 = self.pos[0] + D*np.cos((-ANGLE - self.angle)*np.pi/180)/2
        y2 = self.pos[1] + D*np.sin((-ANGLE - self.angle)*np.pi/180)/2 
        arrSensor = np.array([x2-x1,y2-y1])
        arr1 =  np.mat([light.pos[0] - x1,light.pos[1] - y1])
        arr2 =  np.mat([light.pos[0] - x2,light.pos[1] - y2])
        sl = float(arrSensor*arr1.T)/(np.linalg.norm(arrSensor)*np.linalg.norm(arr1))
        sr = float(arrSensor*arr2.T)/(np.linalg.norm(arrSensor)*np.linalg.norm(arr2))
        return sl,sr
    
    def move(self,motor_value,lights):
        self.pos = self.rect.center
        arr_light_one = np.array(np.array(lights[0].pos)-np.array(self.pos))
        arr_light_two = np.array(np.array(lights[1].pos)-np.array(self.pos))
        # 机器人中心到光源距离
        distance_one = np.linalg.norm(arr_light_one)
        distance_two = np.linalg.norm(arr_light_two)
        if distance_one <= 10 or distance_two <= 10:
            return
        self.vl,self.vr = motor_value
        # print(vl,vr)
        self.angle -= (self.vr - self.vl)/D
        tmp = (self.vr - self.vl)/D
        direct = vec(self.direction()[0],self.direction()[1]).rotate(tmp)
        self.velocity = (self.vl + self.vr)/2 * direct 
        # self.pos += self.velocity
        self.rect = self.rect.move(self.velocity[0], self.velocity[1])
        self.pos = self.rect.center
        # print("id = {id} pos {pos}".format(id=self.id,pos=self.pos))
        # print("id = {id} vel {vel}".format(id=self.id,vel=self.velocity))
        # print("id = {id} angle {angle}".format(id=self.id,angle=self.angle))
        
class GA():
 
    
    def __init__(self,
                n_generations=50,
                population = None,
                selection_type = "sss",
                crossover_type = "single_point",
                mutation_type = "random",
                elite_rate = 1,
                crossover_and_mutation = 50,
                crossover_only = 40,
                mutation_only = 9 ,
                crossover_percent_genes = 100,
                mutation_percent_genes = 100,
                robots = None,
                lights = None):

        self.population = population    
        self.crossover_type = crossover_type
        self.elite_rate = elite_rate
        self.crossover_and_mutation = crossover_and_mutation
        self.crossover_only = crossover_only
        self.mutation_only = mutation_only
        self.crossover_percent_genes = crossover_percent_genes
        self.mutation_percent_genes = mutation_percent_genes
        self.generations = n_generations
        self.robots = robots
        self.lights = lights

        if (crossover_type == "single_point"):
            self.crossover = self.single_point_crossover
        elif (crossover_type == "two_points"):
            self.crossover = self.two_points_crossover
        elif (crossover_type == "uniform"):
            self.crossover = self.uniform_crossover

         # Validating the mutation type: mutation_type
        if (mutation_type == "random"):
            self.mutation = self.random_mutation
        
        self.mutation_type = mutation_type

    
        if (selection_type == "roulette"):
            self.select_parents = self.roulette_wheel_selection
        elif (selection_type == "random"):
            self.select_parents = self.random_selection
        elif (selection_type == "tournament"):
            self.select_parents = self.tournament_selection
        elif (selection_type == "rank"):
            self.select_parents = self.rank_selection
        
        self.K_tournament = K_tournament
        self.best_solution_fitness = []
        self.best_dnas = []


    def run(self):
        for _ in range(self.generations):
            fitness = self.get_fitness()
            best_fit = fitness[np.argmax(fitness)]
            print("Best fitness: ",best_fit)
            # min_dis = 1000/best_fit
            # print("Min distance:",min_dis)
            # if dis<50:
            # print("Most fitted DNA: ", self.population[np.argmax(fitness), :])
            
            cross_robot_idx,parents_cross_only = self.select_parents(fitness,choosed_percentage=self.crossover_only)
            mutation_robot_idx,parents_mutation_only = self.select_parents(fitness, choosed_percentage=self.mutation_only)
            cm_robot_idx,parents_cross_and_mutation = self.select_parents(fitness, choosed_percentage=self.crossover_and_mutation)

            offspring_crossover_only = self.crossover(parents_cross_only,
                                                    offspring_size=(parents_cross_only.shape[0], parents_cross_only.shape[1]))
            
            offspring_cross_and_mutation_c = self.crossover(parents_cross_only,
                                                    offspring_size=(parents_cross_and_mutation.shape[0], parents_cross_and_mutation.shape[1]))

            offspring_mutation_only = self.mutation(parents_mutation_only)
            offspring_cross_and_mutation_m = self.mutation(offspring_cross_and_mutation_c)


            # 基因型与下一代robot个体
            offspring_robots = []

            elite_robot_idx,offspring_elite = self.elite_selection(fitness)
            elite_idx = offspring_elite.shape[0]
            crossover_only_idx = offspring_crossover_only.shape[0] + elite_idx
            mutation_only_idx = offspring_mutation_only.shape[0] + crossover_only_idx

            self.population[0:elite_idx, :] = offspring_elite
            # for i in elite_robot_idx:
            #     offspring_robots.append(self.robots[i])
            offspring_robots.append(self.robots[elite_robot_idx])

            for i in cross_robot_idx:
                offspring_robots.append(self.robots[i])
            for i in mutation_robot_idx:
                offspring_robots.append(self.robots[i])
            for i in cm_robot_idx:
                offspring_robots.append(self.robots[i])
            # offspring_robots.append(self.robots[cross_robot_idx])
            # offspring_robots.append(self.robots[mutation_robot_idx])
            # offspring_robots.append(self.robots[cm_robot_idx])
            self.robots = offspring_robots

            self.population[elite_idx:crossover_only_idx, :] = offspring_crossover_only
            self.population[crossover_only_idx:mutation_only_idx, :] = offspring_mutation_only
            self.population[mutation_only_idx:, :] = offspring_cross_and_mutation_m
 

    def update(self): 
        # for light in self.lights:   
        # light = self.lights[0]
        i=0
        for ann_params  in self.population:
            theta1 = np.matrix(np.reshape(ann_params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
            theta2 = np.matrix(np.reshape(ann_params[hidden_size * (input_size + 1):], (output_size, (hidden_size + 1))))
            X = []
            X.append(self.robots[i].get_angles(self.lights[0]))
            X.append(self.robots[i].get_angles(self.lights[1]))
            X = np.array(X).reshape(-1)
            motor_value = np.array(forward_propagate(np.matrix(X),theta1,theta2))[0]  
            self.robots[i].move(motor_value,self.lights)
            i+=1
            

    def get_fitness(self):
        # print("before",self.robots[0].velocity)
        self.update()
        # print("after",self.robots[0].velocity)
        # print("Robots ",len(self.robots))
        fitness = []
        for robot in self.robots:
            arr_light_one = np.array(np.array(self.lights[0].pos)-np.array(robot.pos))
            arr_light_two = np.array(np.array(self.lights[1].pos)-np.array(robot.pos))
            # 机器人中心到光源距离
            distance_one = np.linalg.norm(arr_light_one)
            distance_two = np.linalg.norm(arr_light_two)
            # if distance_one<distance_two:
            #     fitness.append(1000/distance_one+1)
            # else:
            #     fitness.append(0.01)
            distance = 1000/distance_one +1 + distance_two
            fitness.append(distance)
        fitness = np.array(fitness)
        self.best_solution_fitness.append(np.max(fitness))
        self.best_dnas.append(self.population[np.argmax(fitness), :])
        return fitness

    def best_dna(self):
        idx = np.argmax(self.best_solution_fitness)
        print("Best dna at {idx} generation".format(idx=idx))
        print("Best dna :",self.best_dnas[idx])


    def single_point_crossover(self, parents, offspring_size):
        offspring = np.empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = np.random.randint(low=0, high=parents.shape[1], size=1)[0]

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def two_points_crossover(self, parents, offspring_size):
        offspring = np.empty(offspring_size)
        if (parents.shape[1] == 1): # If the chromosome has only a single gene. In this case, this gene is copied from the second parent.
            crossover_point1 = 0
        else:
            crossover_point1 = np.random.randint(low=0, high=np.ceil(parents.shape[1]/2 + 1), size=1)[0]

        crossover_point2 = crossover_point1 + int(parents.shape[1]/2) # The second point must always be greater than the first point.

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]
            # The genes from the beginning of the chromosome up to the first point are copied from the first parent.
            offspring[k, 0:crossover_point1] = parents[parent1_idx, 0:crossover_point1]
            # The genes from the second point up to the end of the chromosome are copied from the first parent.
            offspring[k, crossover_point2:] = parents[parent1_idx, crossover_point2:]
            # The genes between the 2 points are copied from the second parent.
            offspring[k, crossover_point1:crossover_point2] = parents[parent2_idx, crossover_point1:crossover_point2]
        return offspring

    def uniform_crossover(self, parents, offspring_size):
        offspring = np.empty(offspring_size)
        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]

            genes_source = np.random.randint(low=0, high=2, size=offspring_size[1])
            for gene_idx in range(offspring_size[1]):
                if (genes_source[gene_idx] == 0):
                    # The gene will be copied from the first parent if the current gene index is 0.
                    offspring[k, gene_idx] = parents[parent1_idx, gene_idx]
                elif (genes_source[gene_idx] == 1):
                    # The gene will be copied from the second parent if the current gene index is 1.
                    offspring[k, gene_idx] = parents[parent2_idx, gene_idx]
        return offspring


    def random_mutation(self, offspring):
        mutation_num_genes = np.uint32((self.mutation_percent_genes*offspring.shape[1])/100)
        if mutation_num_genes == 0:
            mutation_num_genes = 1
        mutation_indices = np.array(random.sample(range(0, offspring.shape[1]), mutation_num_genes))
        # Random mutation changes a single gene in each offspring randomly.
        for idx in range(offspring.shape[0]):
            # The random value to be added to the gene.
            # random_value = np.random.uniform(-2, 2, 1)
            random_value =  std * np.random.randint(-2,2)
            offspring[idx, mutation_indices] = offspring[idx, mutation_indices] + random_value
        return offspring

    def elite_selection(self, fitness):
        num_parents = np.uint32((self.elite_rate*self.population.shape[0])/100)
        if num_parents == 0:
            num_parents = 1
        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        parents = np.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[fitness_sorted[parent_num], :]

        robot_idx = fitness_sorted[parent_num]
        return robot_idx,parents
        
    def rank_selection(self, fitness,choosed_percentage):
        robot_idxs = []
        num_parents = np.uint32((choosed_percentage*self.population.shape[0])/100)
        if num_parents == 0:
            num_parents = 1
        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = np.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[fitness_sorted[parent_num], :]
            robot_idxs.append(fitness_sorted[parent_num])
        robot_idxs = np.array(robot_idxs).reshape(-1)
        return robot_idxs,parents
    
    def random_selection(self, fitness, choosed_percentage):
        robot_idxs = []
        num_parents = np.uint32((choosed_percentage*self.population.shape[0])/100)
        if num_parents == 0:
            num_parents = 1
        parents = np.empty((num_parents, self.population.shape[1]))

        rand_indices = np.random.randint(low=0.0, high=fitness.shape[0], size=num_parents)

        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[rand_indices[parent_num], :]
            robot_idxs.append(rand_indices[parent_num])
        robot_idxs = np.array(robot_idxs).reshape(-1)
        return robot_idxs,parents
    
    def tournament_selection(self, fitness, choosed_percentage):
        robot_idxs = []
        num_parents = np.uint32((choosed_percentage*self.population.shape[0])/100)
        if num_parents == 0:
            num_parents = 1
        parents = np.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            rand_indices = np.random.randint(low=0.0, high=len(fitness), size=self.K_tournament)
            K_fitnesses = fitness[rand_indices]
            selected_parent_idx = np.where(K_fitnesses == np.max(K_fitnesses))[0][0]
            parents[parent_num, :] = self.population[rand_indices[selected_parent_idx], :]
            robot_idxs.append(rand_indices[selected_parent_idx])
        robot_idxs = np.array(robot_idxs).reshape(-1)
        return robot_idxs,parents

    def roulette_wheel_selection(self, fitness, choosed_percentage):
        robot_idxs = []
        num_parents = np.uint32((choosed_percentage*self.population.shape[0])/100)
        if num_parents == 0:
            num_parents = 1
        fitness_sum = np.sum(fitness)
        probs = fitness / fitness_sum
        probs_start = np.zeros(probs.shape, dtype=np.float) # An array holding the start values of the ranges of probabilities.
        probs_end = np.zeros(probs.shape, dtype=np.float) # An array holding the end values of the ranges of probabilities.
        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = np.where(probs == np.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = np.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            rand_prob = np.random.rand()
            for idx in range(probs.shape[0]):
                if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                    robot_idxs.append(idx)
                    parents[parent_num, :] = self.population[idx, :]
                    break
        robot_idxs = np.array(robot_idxs)
        return robot_idxs,parents

    def plot_result(self):
        plt.figure()
        plt.plot(self.best_solution_fitness)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.savefig("fitness.jpg")
        plt.show()

    def save(self, filename):
        self.robots=[]
        with open(filename + ".pkl", 'wb') as file:
            pickle.dump(self, file)
   

if __name__ == "__main__":

    robots = []
    for i in range(pop_size):
        robots.append(Robot(i))
    lights = []
    # for j in range(0,2):
    #     x = random.randrange(30, 800 - 30)
    #     y = random.randrange(30, 600 - 30)
    #     light_pos = np.array([x,y])
    #     light = Light(light_pos,j)
    #     lights.append(light)
    light1 = Light(np.array([0,0]),0)
    lights.append(light1)
    light2 = Light(np.array([800,600]),1)
    lights.append(light2)
    n_generations = 100
    # population = std*np.random.randn(pop_size,gene_size)
    population = load(filename="init_population")
    selection_type = "roulette" # roulette rank random tournament
    crossover_type = "single_point" # single_point two_points uniform
    mutation_type = "random"
    elite_rate = 1
    crossover_and_mutation = 80
    crossover_only = 15
    mutation_only = 4 
    crossover_percent_genes = 100
    mutation_percent_genes = 10
    ga = GA(
        n_generations = n_generations,
        population = population,
        selection_type = selection_type,
        crossover_type = crossover_type,
        mutation_type = mutation_type,
        elite_rate = elite_rate,
        crossover_and_mutation = crossover_and_mutation,
        crossover_only = crossover_only,
        mutation_only = mutation_only,
        crossover_percent_genes = crossover_percent_genes,
        mutation_percent_genes = mutation_percent_genes,
        robots = robots,
        lights = lights)
    ga.run()
    filename = "genetic"
    ga.save(filename=filename)
    loaded_ga_instance = load(filename=filename)
    loaded_ga_instance.plot_result()
    loaded_ga_instance.best_dna()
