package project;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class Knapsack {

    private static final int[] WEIGHTS = {2, 5, 7, 3, 1, 5, 8};
    private static final int[] VALUES = {25, 300, 200, 600, 100, 35, 300};
    private static final int CAPACITY = 10;
    private static final int POPULATION_SIZE = 100;
    private static final double MUTATION_RATE = 0.01;
    private static final int NUM_GENERATIONS = 1000;

    private static final ArrayList<Boolean> isCorrect = new ArrayList<>();

    private static final Random random = new Random();

    public static void main(String[] args) {
        double averageSolutionTime = 0;
        double averageSolutionGeneration = 0;
        int numSolutions = 15;
        for (int i = 0; i < numSolutions; i++) {
            long startTime = System.currentTimeMillis();
            int generation = evolve();
            long endTime = System.currentTimeMillis();

            System.out.println("Solution found in " + generation + " generations.");
            System.out.println("Time taken: " + (endTime - startTime) + " milliseconds.");
            System.out.println();
            averageSolutionTime += (endTime - startTime);
            averageSolutionGeneration += generation;
        }

        double correctCount = isCorrect.stream().filter(Boolean::booleanValue).count();
        double incorrectCount = isCorrect.size() - correctCount;
        double accuracyRate = correctCount / numSolutions * 100;

        System.out.println("Average generation: " + (averageSolutionGeneration / numSolutions) + " generations.");
        System.out.println("Average time taken: " + (averageSolutionTime / numSolutions) + " milliseconds.");
        System.out.println();
        System.out.println("Correct solutions: " + correctCount);
        System.out.println("Incorrect solutions: " + incorrectCount);
        System.out.printf("Average accuracy rate: %.2f%%\n", accuracyRate);

    }

    private static int evolve() {
        int[][] population = new int[POPULATION_SIZE][WEIGHTS.length];
        double[] fitness = new double[POPULATION_SIZE];

        // Initialize population
        for (int i = 0; i < POPULATION_SIZE; i++) {
            population[i] = generateRandomSolution();
        }

        int generation = 0;
        while (generation < NUM_GENERATIONS) {
            for (int i = 0; i < POPULATION_SIZE; i++) {
                fitness[i] = calculateFitness(population[i]);
            }

            int[][] newPopulation = new int[POPULATION_SIZE][WEIGHTS.length];
            for (int i = 0; i < POPULATION_SIZE; i++) {
                int[] parent1 = selectParent(population, fitness);
                int[] parent2 = selectParent(population, fitness);
                int[] child = crossover(parent1, parent2);
                mutate(child);
                newPopulation[i] = child;
            }
            population = newPopulation;

            generation++;
        }

        int bestIndividualIndex = 0;
        for (int i = 1; i < POPULATION_SIZE; i++) {
            if (fitness[i] > fitness[bestIndividualIndex]) {
                bestIndividualIndex = i;
            }
        }

        System.out.print("Best solution: ");
        System.out.println(Arrays.toString(population[bestIndividualIndex]));
        System.out.println("Total value: " + calculateFitness(population[bestIndividualIndex]));

        // The correct solution varies depending on each weight and value list.
        if (Arrays.equals(population[bestIndividualIndex], new int[]{0, 1, 0, 1, 1, 0, 0})) {
            System.out.println("Solution is correct :)");
            isCorrect.add(true);
        } else {
            System.out.println("Solution is incorrect :(");
            isCorrect.add(false);
        }

        return generation;
    }

    private static int[] generateRandomSolution() {
        int[] solution = new int[WEIGHTS.length];
        for (int i = 0; i < WEIGHTS.length; i++) {
            solution[i] = random.nextBoolean() ? 1 : 0;
        }
        return solution;
    }

    private static double calculateFitness(int[] candidate) {
        int totalWeight = 0;
        int totalValue = 0;
        for (int i = 0; i < candidate.length; i++) {
            if (candidate[i] == 1) {
                totalWeight += WEIGHTS[i];
                totalValue += VALUES[i];
            }
        }
        if (totalWeight > CAPACITY) {
            return 0;
        }
        return totalValue;
    }

    private static int[] selectParent(int[][] population, double[] fitness) {
        double totalFitness = Arrays.stream(fitness).sum();
        double randomFitness = random.nextDouble() * totalFitness;
        double sum = 0;
        for (int i = 0; i < POPULATION_SIZE; i++) {
            sum += fitness[i];
            if (sum >= randomFitness) {
                return population[i];
            }
        }
        return population[POPULATION_SIZE - 1];
    }

    private static int[] crossover(int[] parent1, int[] parent2) {
        int[] child = new int[WEIGHTS.length];
        int midpoint = random.nextInt(WEIGHTS.length);
        for (int i = 0; i < WEIGHTS.length; i++) {
            if (i < midpoint) {
                child[i] = parent1[i];
            } else {
                child[i] = parent2[i];
            }
        }
        return child;
    }

    private static void mutate(int[] child) {
        for (int i = 0; i < WEIGHTS.length; i++) {
            if (random.nextDouble() < MUTATION_RATE) {
                child[i] = 1 - child[i];
            }
        }
    }
}
