# The evolution of a controller’s behavior in optimally solving OpenAI’s CartPole task.
## Abstract:
Evolving neural networks is a form of learning used to evolve a neural network that controls an agent. To train the neural network, various methods can be employed, one of which is genetic algorithms which this report explores. This report investigates how the agent’s behavior and way of solving the cartpole task evolve as the neural network is trained. It also examines how the genetic algorithm is able to optimize the neural network making it able to output optimal actions given any set of valid inputs.

The research question that this report aims to solve is, therefore, **“How does the behavior of a controller evolve so that it can optimally solve a given task?”**. This report first examines how the maximum fitness of genotypes evolves over generations in the genetic algorithm. It then looks at how the behavior of the agent evolves, through analyzing plots of observations within the cartpole’s environment, using the best genotype at generations 0, 50, and 100. 

The results from these experiments are then discussed, before concluding that the GA increases the maximum fitness using large jumps in earlier generations that then quickly stabilize at the maximum max-fitness after the halfway point. Results also showed that the best genotype at generation 50 led to a failure-avoidance approach to solving the task, while the genotype at generation 100 led to a truly optimal approach to solving the task. 

## Introduction:
This report aims to investigate how an agent evolves so that it is able to optimally solve a given task. The one being examined in this case is the cartpole task, an OpenAI gym environment that is trained with the goal of balancing a pole on top of a moving cart. 

A genetic algorithm is used to train the agent’s neural network. This allows us to look at how a population of genotypes improves so that they can provide the neural network with the ideal weights and bias to help it optimally solve the cartpole task. Using a genetic algorithm also allows for a detailed analysis of how the maximum max-fitness evolves over generations. Specific factors such as the population’s diversity and accuracy in solving the task at different generations are also looked into. 

As this report also focuses on the behavior of the agent while solving the task, plots of how the environment’s observations change over time are given and are then deeply analyzed. This also allows us to observe how the agent outputs actions that respond to specific observation inputs.

Therefore, the research question being investigated is **“How does the behavior of a controller evolve so that it can optimally solve a given task?"**.
## Methods:
The cartpole task, which this report aims to find a solution for, is defined as the following:
“*A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.* <sup>1</sup>

<img width="454" alt="image" src="https://user-images.githubusercontent.com/84683922/184545572-d0f03741-93fc-4a2a-91eb-936b97ad0780.png">

To act on applying a force as given above, the cartpole’s environment supplies us with a `step()` function which takes as input the discrete value of either a 0 (moving the cart to the left) or a 1 (moving the cart to the right). It then returns the four following items:
* Observations, which consist of four values:
  * The position of the cart.
  * The velocity of the cart.
  * The angle of the pole.
  * The rotation rate of the pole.
* Reward
* Done, a boolean variable, used to indicate the episode’s termination when set to true.
* Info, a dictionary that provides diagnostic information for debugging, is not used here.

The goal is therefore to develop an agent that, using a neural network, can take in the four observations as input and generate an action that maximizes the reward returned. With the GA using genotypes, a way to map a genotype to weights and biases which the agent’s neural network can use was also needed. Therefore, the agent class used the following mapping:
```
weight_indexes = num_input * num_output
bias_indexes = num_input * num_output + num_output
weights = gene[0:weight_indexes], reshaped to a 2D matrix of shape (num_output, num_input)
bias = gene[weight_indexes:bias_indexes], reshaped to a 1D array of shape (num_output)
```

Important things to note:
* The number of inputs, in this case, is 4 as the neural network takes 4 observations.
* The number of outputs is 1 as the neural network outputs 1 action. 
* Genes in a genotype consist of floats in the range of -3 and 3. 
* Genotypes consist of 5 genes, with their size being calculated through:
  * `num_input * num_output + num_output`

The agent class also has a `forward()` function used for forward propagation, the function implements the following formula: $$y=(X*W^T+b)$$

where X represents the 4 inputs (observations), WT represents the transposed weights, and b represents the bias.

A threshold on the value of y is then used to output the action to be performed, with values greater than 0 resulting in a 1 (moving the cart to the right), and values less than or equal to 0 resulting in a 0 (moving the cart to the left). 

A `run_episode()` function which takes in an environment and an agent as parameters was implemented with the purpose of getting an action from the agent, using the environment’s `step()` function with that action, and doing that repeatedly until the “done” boolean variable is set to true. The function then returns a list of all generated observations, in addition to the total reward accumulated throughout that episode.

With the `run_episode()` function and the agent being fully implemented, the next task was to optimize the controller and enable it to reliably solve the cartpole task. To do this, a GA consisting of four parts was developed.

The four parts of the genetic algorithm:
* Creating the initial population:
  * A population size of 15 was used.
  * Initially, all genes consist of zeros.
    * This allows for a better way of analyzing and visualizing how the GA optimizes the agent over time/generations.
* Fitness function:
  * Creates a fitness array of the same size as the genotype population.
  * Uses the agent with each genotype and assigns the fitness of each genotype in the fitness array as the total reward returned from the run_episode() function.
* Mutation function:
  * Adds gaussian noise from a normal distribution with mean 0 and standard deviation 0.1 to each gene in each genotype.
  * The gaussian noise added/subtracted is the same for each gene in a single genotype but is randomized between genotypes.
  * Thresholding is then performed on each gene in each genotype, with values greater than 3 being set to 3, and values less than -3 set to -3.
* Selection function:
  * Takes four parameters:
    * Current genotype population, current population’s fitness, previous genotype population, previous population’s fitness.
  * Creates a new genotype population keeping only genotypes that have a better fitness than their counterpart in the other population, either current or previous.

These four parts are then all combined together to form the GA which allows us to optimize the genotypes so that the agent will be able to provide actions that maximize the reward from the environment’s `step()` function. The GA’s pseudocode is given below:
```
ga(generations, pop_size, mutation_mean, mutation_std, num_obs, 
   num_actions)
   agent = Agent(num_obs, num_actions)
   gene_pop = Matrix of zeros, shape (pop_size, agent.num_genes)
   fitness = fitness_function(env, agent, gene_pop)
   prev_fitness = fitness
   prev_gene_pop = gene_pop
   for i = 0 to range(generations)
       gene_pop = mutation_function(prev_gene_pop, mutation_mean, 
                                    mutation_std)
       fitness = fitness_function(env, agent, gene_pop)
       gene_pop = select_genes(prev_gene_pop, gene_pop, 
                               prev_fitness, fitness)
       prev_fitness = fitness_function(env, agent, gene_pop)
       prev_gene_pop = gene_pop
       return gene_pop
```

## Results:
Using the following parameters:
* Generations = 100
* Population Size = 15
* Mutation Mean = 0
* Mutation Standard Deviation = 0.1
* Number of Observations = 4
* Number of Actions = 1

The GA was run ten times, returning a list of the maximum fitness (fitness of the fittest individual) over all 100 generations.

The lists returned from the ten trials were then used to get the mean and standard error of the maximum fitness at each generation. These values were then used to generate the below plot. The black line represents the mean of the maximum fitness, while the gray area above and below the line shows the standard error.

<img width="454" alt="image" src="https://user-images.githubusercontent.com/84683922/184545857-06691fe2-8274-4925-b18e-83d4f4280599.png">

Through this plot, it can be inferred that the maxim fitness uses big jumps to increase the maximum fitness between generations 0 and 20, before then slowly increasing it through similarly sized increases and decreases in the maximum fitness between generations 20 and 40. The maximum fitness then stabilizes at a max fitness of 500, the maximum max-fitness that can be obtained.

Another run of the GA was used to record the fittest genotype at generations 0, 50, and 100. The values of these genotypes are given below:

<img width="805" alt="image" src="https://user-images.githubusercontent.com/84683922/184546030-75ebcb2f-652e-448e-9d14-b1099971ff05.png">

These values were then used to generate 4 plots of the observations per genotype (12 plots total).

### Fittest Genotype at Generation #0 Plots:

<img width="814" alt="image" src="https://user-images.githubusercontent.com/84683922/184545901-efc935b7-d41f-4261-ae06-f14c608a7926.png">

We can see from Figure 3c that the episode has ended early due to the pole’s angle moving by 15 degrees from the starting point. The cart’s position also never stabilized as can be seen from Figure 3a, indicating that a way to balance the pole has not yet been found by the agent.

### Fittest Genotype at Generation #50 Plots:

<img width="811" alt="image" src="https://user-images.githubusercontent.com/84683922/184545921-f60669e1-0762-459d-ad0e-f2f788894a06.png">

### Fittest Genotype at Generation #100 Plots:

<img width="811" alt="image" src="https://user-images.githubusercontent.com/84683922/184545936-6968cf28-ca66-4dae-a91b-fadd54ee195c.png">

The previous 8 plots show the cartpole task being solved (indicated by the plot’s 500 time-steps) by the agent when tuned with the fittest genotypes from generations 50 and 100. The figures 5a, 5b, 5c, and 5d all showed noticeable improvements over figures 4a, 4b, 4c, and 4d respectively. This means that although the max reward can be reached at an early point, further optimization of the controller leads to noticeable behavioral improvements in relation to how the agent approaches solving the cartpole task. 

The two main improvements can be seen in the cart’s position and the pole’s angle. These two observations are what ultimately decide when the task fails and so optimizing them should be the main focus of the agent. We can see that the cart’s position in Figure 5a is much more stable and consistent than the cart’s position in Figure 4a, while also having a position that is much closer to the center of the environment. 

The pole’s angle also quickly stabilized at an angle of 0 degrees in Figure 5c, unlike Figure 4c which shows the pole’s angle very slowly getting closer to 0 degrees. Additionally, even when the pole angle does eventually reach 0 degrees in Figure 4c, the agent is unable to maintain that angle for the remaining time before the episode ends.

Visualizations for the behavior of each genotype when running an episode were also created and saved as a video, they are available [here](https://youtube.com/playlist?list=PLEwYjESKyptHjZQZVEe_I5a8nO0yq9_8Q).

To get the accuracy of the agent using each one of these 3 genotypes, a function was created that would run 10,000 episodes, count the number of successful episodes (ones that return a total reward of 500), and divide that by 10,000. The following results were obtained:

<img width="517" alt="image" src="https://user-images.githubusercontent.com/84683922/184546010-11f19361-83ab-43a7-8483-ff7cc8ae17c7.png">

These results further prove the discoveries made earlier indicating that the maximum total reward can be obtained at an early stage (after 50 generations), while further optimizations after that mark mainly aim to improve and perfect the behavior of the controller.

## Analysis & Discussion:
One of the main reasons behind the large increases in the maximum fitness within the population in Figure 2 is the relatively large amount of gaussian noise added as part of the mutation function. The gaussian noise used within that function had a mean of 0 and a standard deviation of 0.1, this allowed mutated genotypes to vary during the earlier stages of training, leading to quicker results in terms of max fitness increases.

However, the use of a large amount of mutation rate also comes with the downside of causing the algorithm to then only very slowly reach the optimal values for a genotype. This downside isn’t hugely impactful in this experiment as even values which are close to the optimal values give great results as Table 2 shows, with the fittest genotype at generation #50 having a great accuracy of 99.93%. 

An interesting characteristic of the observation plots for the fittest genotype at generation #100 (Figures 5a, 5b, 5c, 5d) is that the agent’s actions are able to stabilize the cart and pole at an early point, after which their position is maintained with very consistent movements/changes up until the episode ends. This is unlike the behavior showcased in Figures 4a, 4b, 4c, and 4d, where the changes and movements seem to only be combative actions to prevent the episode from ending at an early point. 

This can particularly be noticed at around generation #450, in which a change is made to stabilize the cart and prevent it from moving out of the environment's limits. This action shows that the agent outputs actions which are aware of the environment space and ways of failure which need to be avoided.

The GA defined to solve this problem is a variation of the standard generational GA. Sharing the characteristic of there being P number of individuals, where P is the population size, being created and mutated each generation. However, unlike a standard generational algorithm that replaces all parent individuals with their offspring, the GA defined here only replaces parents with fitter offsprings. This results in it still exhibiting elitism as the fitter parents are carried over to the next generation.<sup>3</sup> 

As the given GA mutates and calculates the fitness values of all new individuals, this results in it being computationally inefficient when compared to other forms of GAs which only use two parents to generate a new offspring each generation. Additionally, although it might seem that the mean fitness can’t decrease between generations as the fitness of every gene is compared against a new one with only the fitter gene being kept, the use of only a single episode to calculate the fitness means that less fit genes might be chosen over fitter genes if they perform better in that single episode. The effect this has can especially be seen in Figure 2, in the generation range of 10 to 40, where this type of error is most prominent.

The GA used here also lacks a crossover function which could make the search process for finding the most optimal values for genotypes faster. Using one which crosses values from the fittest genotype onto less fit individuals could result in the population being more diverse and therefore able to explore more solutions. The use of one, however, also introduces the possibility of converging towards a local minimum in the solution space.<sup>4</sup> A way to prevent this is through the use of a mutation function which could make getting out of weak local minima possible.

A future area of research for this report is trying out different hyperparameters (population size, mutation mean, mutation standard deviation, crossover probability if it were to be added) for the GA, and testing out the effects they may have on how the agent’s behavior evolves. Additionally, further research on more complex neural networks and ways to map genotypes to weights and biases could offer great insight into how the performance of the current agent could be improved.

## Conclusion:
In conclusion, this report showcased how the agent’s behavior evolves, exploring things such as how the maximum fitness evolves over generations, and how the agent performs using the fittest genotype at generations 0, 50, and 100. 

The findings from these experiments show that GA’s mutation function allows for large increases in the maximum fitness, which are attributed to increases in the population’s genetic diversity, during the earlier generations. These optimizations then stabilize at the maximum max-fitness near the halfway point, with further optimizations after that point allowing the agent to give the optimal actions and exhibit the ideal behavior whilst solving the cartpole task.

How the agent performs when using the fittest genotype at generation #0 was investigated, with results indicating that the agent failed in stabilizing the pole as a way to balance the pole has not yet been found by the agent.

A comparison of how the agent performs using the fittest genotype at generation #50 vs #100 was also made. Results showed that using the former led to the agent being combative in relation to how it avoids failure, while the latter led to a more stable and consistent approach in solving the cartpole task.

To build upon the conclusions reached in this report and to enable further exploration, future research areas may target the effects of varying the GA’s hyperparameters on the agent’s optimization, the introduction of a crossover function, as well as exploring different neural network architectures and genotype mappings.

## Bibliography:
1 “OpenAI Gym: The CartPole-v0 Environment.” Gym.openai.com, 2017, gym.openai.com/envs/CartPole-v0/.

2 “Introducing CartPole-V1.” Packtpub.com, 2019, subscription.packtpub.com/book/big-data-and-business-intelligence/9781789345803/8/ch08lvl1sec46/introducing-cartpole-v1.

3 Jenkins, Alison, et al. Variations of Genetic Algorithms. 1 Nov. 2019.

4 Davis, Harry. “How Can We Avoid Local Minima in Genetic Algorithm? – QuickAdviser.” Quick-Adviser.com, 12 June 2020, quick-adviser.com/how-can-we-avoid-local-minima-in-genetic-algorithm/. Accessed 21 Apr. 2022.

## Appendix:
### Full Code:
See Report_3.ipynb file.
















