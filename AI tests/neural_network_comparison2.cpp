/**
 * Neural Network Training Comparison #2: Smarter Quaylyn's Law Implementation
 * 
 * Problem with v1: Testing N sections for each of 58 parameters = 522 evaluations/iteration
 * 
 * Solution: POPULATION-BASED ELIMINATION
 * - Maintain a population of N networks (candidates)
 * - Evaluate all networks on same data
 * - Eliminate worst 1/3 of population (Quaylyn's optimal rate)
 * - Mutate survivors to fill population
 * - Repeat until convergence
 * 
 * This captures Quaylyn's Law (eliminate worst, don't commit to "best") while being
 * computationally efficient (N evaluations per generation, not N × params)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <numeric>

// ============================================================================
// CONFIGURATION
// ============================================================================

constexpr int HIDDEN_NEURONS = 8;          // Hidden layer size
constexpr int INPUT_SIZE = 4;              // Input features
constexpr int OUTPUT_SIZE = 2;             // Output classes
constexpr int TRAINING_SAMPLES = 1000;     // Training set size
constexpr int TEST_SAMPLES = 200;          // Test set size
constexpr int MAX_EPOCHS = 500;            // Maximum training epochs
constexpr double LEARNING_RATE = 0.1;      // For gradient descent
constexpr double TARGET_ACCURACY = 0.95;   // Stop when reached
constexpr int NUM_TRIALS = 10;             // Repeated trials for averaging

// Quaylyn's Law population settings
// N=3: Eliminate 1/3 (33%) per generation
constexpr int POPULATION_SIZE_N3 = 9;      // Number of candidate networks
constexpr int ELIMINATION_COUNT_N3 = 3;    // Eliminate 3/9 = 33%

// N=9: Eliminate 1/9 (11.1%) per generation  
constexpr int POPULATION_SIZE_N9 = 9;      // Number of candidate networks
constexpr int ELIMINATION_COUNT_N9 = 1;    // Eliminate 1/9 = 11.1%

constexpr double MUTATION_RATE = 0.3;      // How much to mutate survivors
constexpr double MUTATION_STRENGTH = 0.5;  // Magnitude of mutations

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

class ProgressBar {
private:
    int total;
    int current;
    int barWidth;
    std::string prefix;
    std::chrono::steady_clock::time_point startTime;
    
public:
    ProgressBar(int total, const std::string& prefix = "", int width = 25) 
        : total(total), current(0), barWidth(width), prefix(prefix) {
        startTime = std::chrono::steady_clock::now();
    }
    
    void update(int value) {
        current = value;
        display();
    }
    
    void display() {
        float progress = static_cast<float>(current) / total;
        int filled = static_cast<int>(barWidth * progress);
        
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - startTime).count();
        double eta = (progress > 0) ? (elapsed / progress - elapsed) : 0;
        
        std::cout << "\r  " << prefix;
        std::cout << " [";
        for (int i = 0; i < barWidth; i++) {
            if (i < filled) std::cout << "█";
            else if (i == filled) std::cout << "▓";
            else std::cout << "░";
        }
        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100) << "% ";
        std::cout << current << "/" << total;
        std::cout << " [" << std::fixed << std::setprecision(1) << elapsed << "s";
        if (eta > 0.1 && progress < 1.0) {
            std::cout << "<" << eta << "s";
        }
        std::cout << "]    " << std::flush;
    }
    
    void finish(const std::string& message = "") {
        current = total;
        display();
        if (!message.empty()) {
            std::cout << " " << message;
        }
        std::cout << "\n";
    }
};

double randomDouble(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-std::clamp(x, -500.0, 500.0)));
}

double sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// ============================================================================
// DATA GENERATION
// ============================================================================

struct DataPoint {
    std::vector<double> input;
    std::vector<double> target;
    int label;
};

std::vector<DataPoint> generateDataset(int numSamples, double noiseLevel = 0.1) {
    std::vector<DataPoint> data;
    std::normal_distribution<double> noise(0.0, noiseLevel);
    
    for (int i = 0; i < numSamples; i++) {
        DataPoint dp;
        dp.input.resize(INPUT_SIZE);
        dp.target.resize(OUTPUT_SIZE, 0.0);
        
        for (int j = 0; j < INPUT_SIZE; j++) {
            dp.input[j] = randomDouble(0.0, 1.0);
        }
        
        // Complex non-linear classification: (x0 XOR x1) AND (x2 > x3)
        bool xor_result = (dp.input[0] > 0.5) != (dp.input[1] > 0.5);
        bool comp_result = dp.input[2] > dp.input[3];
        
        dp.label = (xor_result && comp_result) ? 1 : 0;
        dp.target[dp.label] = 1.0;
        
        for (int j = 0; j < INPUT_SIZE; j++) {
            dp.input[j] += noise(rng);
            dp.input[j] = std::clamp(dp.input[j], 0.0, 1.0);
        }
        
        data.push_back(dp);
    }
    
    return data;
}

// ============================================================================
// NEURAL NETWORK
// ============================================================================

struct NeuralNetwork {
    std::vector<std::vector<double>> weights_ih;
    std::vector<std::vector<double>> weights_ho;
    std::vector<double> bias_h;
    std::vector<double> bias_o;
    std::vector<double> hidden_raw, hidden_act, output_raw, output_act;
    
    NeuralNetwork() {
        weights_ih.resize(HIDDEN_NEURONS, std::vector<double>(INPUT_SIZE));
        weights_ho.resize(OUTPUT_SIZE, std::vector<double>(HIDDEN_NEURONS));
        bias_h.resize(HIDDEN_NEURONS);
        bias_o.resize(OUTPUT_SIZE);
        hidden_raw.resize(HIDDEN_NEURONS);
        hidden_act.resize(HIDDEN_NEURONS);
        output_raw.resize(OUTPUT_SIZE);
        output_act.resize(OUTPUT_SIZE);
        randomizeWeights();
    }
    
    void randomizeWeights() {
        double range = std::sqrt(6.0 / (INPUT_SIZE + HIDDEN_NEURONS));
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                weights_ih[i][j] = randomDouble(-range, range);
            }
            bias_h[i] = randomDouble(-0.1, 0.1);
        }
        
        range = std::sqrt(6.0 / (HIDDEN_NEURONS + OUTPUT_SIZE));
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                weights_ho[i][j] = randomDouble(-range, range);
            }
            bias_o[i] = randomDouble(-0.1, 0.1);
        }
    }
    
    std::vector<double> forward(const std::vector<double>& input) {
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            hidden_raw[i] = bias_h[i];
            for (int j = 0; j < INPUT_SIZE; j++) {
                hidden_raw[i] += weights_ih[i][j] * input[j];
            }
            hidden_act[i] = sigmoid(hidden_raw[i]);
        }
        
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output_raw[i] = bias_o[i];
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                output_raw[i] += weights_ho[i][j] * hidden_act[j];
            }
            output_act[i] = sigmoid(output_raw[i]);
        }
        
        return output_act;
    }
    
    int predict(const std::vector<double>& input) {
        auto output = forward(input);
        return (output[0] > output[1]) ? 0 : 1;
    }
    
    double calculateLoss(const std::vector<DataPoint>& data) {
        double totalLoss = 0.0;
        for (const auto& dp : data) {
            auto output = forward(dp.input);
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                double diff = output[i] - dp.target[i];
                totalLoss += diff * diff;
            }
        }
        return totalLoss / data.size();
    }
    
    double calculateAccuracy(const std::vector<DataPoint>& data) {
        int correct = 0;
        for (const auto& dp : data) {
            if (predict(dp.input) == dp.label) correct++;
        }
        return static_cast<double>(correct) / data.size();
    }
    
    // Copy weights from another network
    void copyFrom(const NeuralNetwork& other) {
        weights_ih = other.weights_ih;
        weights_ho = other.weights_ho;
        bias_h = other.bias_h;
        bias_o = other.bias_o;
    }
    
    // Mutate weights with given rate and strength
    void mutate(double rate, double strength) {
        std::normal_distribution<double> noise(0.0, strength);
        std::uniform_real_distribution<double> chance(0.0, 1.0);
        
        for (auto& row : weights_ih) {
            for (auto& w : row) {
                if (chance(rng) < rate) w += noise(rng);
            }
        }
        for (auto& b : bias_h) {
            if (chance(rng) < rate) b += noise(rng);
        }
        for (auto& row : weights_ho) {
            for (auto& w : row) {
                if (chance(rng) < rate) w += noise(rng);
            }
        }
        for (auto& b : bias_o) {
            if (chance(rng) < rate) b += noise(rng);
        }
    }
    
    // Crossover: blend weights from two parents
    void crossover(const NeuralNetwork& parent1, const NeuralNetwork& parent2) {
        std::uniform_real_distribution<double> blend(0.0, 1.0);
        
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                double t = blend(rng);
                weights_ih[i][j] = t * parent1.weights_ih[i][j] + (1-t) * parent2.weights_ih[i][j];
            }
            double t = blend(rng);
            bias_h[i] = t * parent1.bias_h[i] + (1-t) * parent2.bias_h[i];
        }
        
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                double t = blend(rng);
                weights_ho[i][j] = t * parent1.weights_ho[i][j] + (1-t) * parent2.weights_ho[i][j];
            }
            double t = blend(rng);
            bias_o[i] = t * parent1.bias_o[i] + (1-t) * parent2.bias_o[i];
        }
    }
};

// ============================================================================
// GRADIENT DESCENT TRAINER (same as v1)
// ============================================================================

struct GradientDescentTrainer {
    void train(NeuralNetwork& nn, const std::vector<DataPoint>& data, double learningRate) {
        std::vector<std::vector<double>> grad_ih(HIDDEN_NEURONS, std::vector<double>(INPUT_SIZE, 0.0));
        std::vector<std::vector<double>> grad_ho(OUTPUT_SIZE, std::vector<double>(HIDDEN_NEURONS, 0.0));
        std::vector<double> grad_bh(HIDDEN_NEURONS, 0.0);
        std::vector<double> grad_bo(OUTPUT_SIZE, 0.0);
        
        for (const auto& dp : data) {
            nn.forward(dp.input);
            
            std::vector<double> output_delta(OUTPUT_SIZE);
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                double error = nn.output_act[i] - dp.target[i];
                output_delta[i] = error * sigmoidDerivative(nn.output_raw[i]);
            }
            
            std::vector<double> hidden_delta(HIDDEN_NEURONS);
            for (int i = 0; i < HIDDEN_NEURONS; i++) {
                double error = 0.0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    error += output_delta[j] * nn.weights_ho[j][i];
                }
                hidden_delta[i] = error * sigmoidDerivative(nn.hidden_raw[i]);
            }
            
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                for (int j = 0; j < HIDDEN_NEURONS; j++) {
                    grad_ho[i][j] += output_delta[i] * nn.hidden_act[j];
                }
                grad_bo[i] += output_delta[i];
            }
            
            for (int i = 0; i < HIDDEN_NEURONS; i++) {
                for (int j = 0; j < INPUT_SIZE; j++) {
                    grad_ih[i][j] += hidden_delta[i] * dp.input[j];
                }
                grad_bh[i] += hidden_delta[i];
            }
        }
        
        double scale = learningRate / data.size();
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                nn.weights_ih[i][j] -= scale * grad_ih[i][j];
            }
            nn.bias_h[i] -= scale * grad_bh[i];
        }
        
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                nn.weights_ho[i][j] -= scale * grad_ho[i][j];
            }
            nn.bias_o[i] -= scale * grad_bo[i];
        }
    }
};

// ============================================================================
// QUAYLYN'S LAW: POPULATION-BASED ELIMINATION
// ============================================================================

struct QuaylynPopulationTrainer {
    int populationSize;
    int eliminationCount;
    double mutationRate;
    double mutationStrength;
    
    QuaylynPopulationTrainer(int popSize, 
                              int elimCount,
                              double mutRate = MUTATION_RATE,
                              double mutStrength = MUTATION_STRENGTH)
        : populationSize(popSize), eliminationCount(elimCount), 
          mutationRate(mutRate), mutationStrength(mutStrength) {}
    
    NeuralNetwork train(const std::vector<DataPoint>& trainData,
                        const std::vector<DataPoint>& valData,
                        int maxGenerations,
                        ProgressBar* progress = nullptr) {
        
        // Initialize population
        std::vector<NeuralNetwork> population(populationSize);
        
        NeuralNetwork bestNetwork;
        double bestLoss = 1e9;
        
        for (int gen = 0; gen < maxGenerations; gen++) {
            // Evaluate all networks
            std::vector<std::pair<double, int>> scores;  // (loss, index)
            for (int i = 0; i < populationSize; i++) {
                double loss = population[i].calculateLoss(valData);
                scores.push_back({loss, i});
                
                if (loss < bestLoss) {
                    bestLoss = loss;
                    bestNetwork.copyFrom(population[i]);
                }
            }
            
            // Sort by loss (best first)
            std::sort(scores.begin(), scores.end());
            
            // QUAYLYN'S LAW: Eliminate worst 1/3 (don't commit to "best", just remove worst)
            std::vector<int> survivors;
            for (int i = 0; i < populationSize - eliminationCount; i++) {
                survivors.push_back(scores[i].second);
            }
            
            // Create new generation
            std::vector<NeuralNetwork> newPopulation(populationSize);
            
            // Keep survivors unchanged
            for (size_t i = 0; i < survivors.size(); i++) {
                newPopulation[i].copyFrom(population[survivors[i]]);
            }
            
            // Fill eliminated slots with crossover + mutation of survivors
            std::uniform_int_distribution<int> parentDist(0, survivors.size() - 1);
            for (int i = survivors.size(); i < populationSize; i++) {
                int p1 = parentDist(rng);
                int p2 = parentDist(rng);
                while (p2 == p1 && survivors.size() > 1) p2 = parentDist(rng);
                
                newPopulation[i].crossover(population[survivors[p1]], 
                                           population[survivors[p2]]);
                newPopulation[i].mutate(mutationRate, mutationStrength);
            }
            
            // Also mutate some survivors (except best) to maintain diversity
            for (size_t i = 1; i < survivors.size(); i++) {
                if (randomDouble(0, 1) < 0.3) {
                    newPopulation[i].mutate(mutationRate * 0.5, mutationStrength * 0.5);
                }
            }
            
            population = std::move(newPopulation);
            
            if (progress && gen % 5 == 0) {
                progress->update(gen);
            }
        }
        
        if (progress) progress->update(maxGenerations);
        
        return bestNetwork;
    }
};

// ============================================================================
// TRAINING METRICS
// ============================================================================

struct TrainingMetrics {
    double trainingTime;
    double finalAccuracy;
    int epochsToConverge;
    double inferenceTime;
    double finalLoss;
    bool converged;
};

// ============================================================================
// RUN TESTS
// ============================================================================

TrainingMetrics trainWithGradientDescent(const std::vector<DataPoint>& trainData,
                                          const std::vector<DataPoint>& testData,
                                          ProgressBar* progress = nullptr) {
    TrainingMetrics metrics;
    NeuralNetwork nn;
    GradientDescentTrainer trainer;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    metrics.epochsToConverge = MAX_EPOCHS;
    metrics.converged = false;
    
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        trainer.train(nn, trainData, LEARNING_RATE);
        
        double accuracy = nn.calculateAccuracy(testData);
        if (accuracy >= TARGET_ACCURACY && !metrics.converged) {
            metrics.epochsToConverge = epoch + 1;
            metrics.converged = true;
        }
        
        if (progress && epoch % 10 == 0) progress->update(epoch);
    }
    
    if (progress) progress->update(MAX_EPOCHS);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    metrics.trainingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    metrics.finalAccuracy = nn.calculateAccuracy(testData);
    metrics.finalLoss = nn.calculateLoss(testData);
    
    auto inferStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) nn.predict(testData[i % testData.size()].input);
    auto inferEnd = std::chrono::high_resolution_clock::now();
    metrics.inferenceTime = std::chrono::duration<double, std::micro>(inferEnd - inferStart).count() / 1000.0;
    
    return metrics;
}

TrainingMetrics trainWithQuaylynPopulation(const std::vector<DataPoint>& trainData,
                                            const std::vector<DataPoint>& testData,
                                            int popSize, int elimCount,
                                            ProgressBar* progress = nullptr) {
    TrainingMetrics metrics;
    QuaylynPopulationTrainer trainer(popSize, elimCount);
    
    // Split for validation
    std::vector<DataPoint> valData(trainData.begin(), trainData.begin() + trainData.size() / 5);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    NeuralNetwork bestNet = trainer.train(trainData, valData, MAX_EPOCHS, progress);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    metrics.trainingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    metrics.finalAccuracy = bestNet.calculateAccuracy(testData);
    metrics.finalLoss = bestNet.calculateLoss(testData);
    metrics.epochsToConverge = MAX_EPOCHS;
    metrics.converged = metrics.finalAccuracy >= TARGET_ACCURACY;
    
    auto inferStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) bestNet.predict(testData[i % testData.size()].input);
    auto inferEnd = std::chrono::high_resolution_clock::now();
    metrics.inferenceTime = std::chrono::duration<double, std::micro>(inferEnd - inferStart).count() / 1000.0;
    
    return metrics;
}

// HYBRID: Elimination first, then backprop to fine-tune
TrainingMetrics trainHybrid(const std::vector<DataPoint>& trainData,
                            const std::vector<DataPoint>& testData,
                            int popSize, int elimCount,
                            ProgressBar* progress = nullptr) {
    TrainingMetrics metrics;
    QuaylynPopulationTrainer popTrainer(popSize, elimCount);
    GradientDescentTrainer gdTrainer;
    
    std::vector<DataPoint> valData(trainData.begin(), trainData.begin() + trainData.size() / 5);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Phase 1: Elimination-based search (half the epochs)
    int elimEpochs = MAX_EPOCHS / 2;
    NeuralNetwork bestNet = popTrainer.train(trainData, valData, elimEpochs, nullptr);
    
    if (progress) progress->update(elimEpochs);
    
    // Phase 2: Fine-tune with backpropagation (other half)
    for (int epoch = 0; epoch < MAX_EPOCHS - elimEpochs; epoch++) {
        gdTrainer.train(bestNet, trainData, LEARNING_RATE);
        if (progress && epoch % 10 == 0) {
            progress->update(elimEpochs + epoch);
        }
    }
    
    if (progress) progress->update(MAX_EPOCHS);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    metrics.trainingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    metrics.finalAccuracy = bestNet.calculateAccuracy(testData);
    metrics.finalLoss = bestNet.calculateLoss(testData);
    metrics.epochsToConverge = MAX_EPOCHS;
    metrics.converged = metrics.finalAccuracy >= TARGET_ACCURACY;
    
    auto inferStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) bestNet.predict(testData[i % testData.size()].input);
    auto inferEnd = std::chrono::high_resolution_clock::now();
    metrics.inferenceTime = std::chrono::duration<double, std::micro>(inferEnd - inferStart).count() / 1000.0;
    
    return metrics;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       NEURAL NETWORK TRAINING #2: POPULATION-BASED QUAYLYN'S LAW                           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Configuration:\n";
    std::cout << "  • Network: " << INPUT_SIZE << " inputs → " << HIDDEN_NEURONS << " hidden → " << OUTPUT_SIZE << " outputs\n";
    std::cout << "  • Training samples: " << TRAINING_SAMPLES << ", Test samples: " << TEST_SAMPLES << "\n";
    std::cout << "  • Max epochs/generations: " << MAX_EPOCHS << "\n";
    std::cout << "  • Gradient descent: learning rate " << LEARNING_RATE << "\n";
    std::cout << "  • N=3 (33% elim): population " << POPULATION_SIZE_N3 << ", eliminate " << ELIMINATION_COUNT_N3 << "\n";
    std::cout << "  • N=9 (11% elim): population " << POPULATION_SIZE_N9 << ", eliminate " << ELIMINATION_COUNT_N9 << "\n";
    std::cout << "  • Mutation: " << (MUTATION_RATE * 100) << "% rate, " << MUTATION_STRENGTH << " strength\n";
    std::cout << "  • Number of trials: " << NUM_TRIALS << "\n\n";
    
    std::vector<TrainingMetrics> gdMetrics, n3Metrics, n9Metrics, n3HybridMetrics, n9HybridMetrics;
    
    std::cout << "Running " << NUM_TRIALS << " trials...\n\n";
    
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        std::cout << "  ╭─ Trial " << (trial + 1) << "/" << NUM_TRIALS << " ───────────────────────────────────────────────────────────────╮\n";
        
        auto trainData = generateDataset(TRAINING_SAMPLES, 0.1);
        auto testData = generateDataset(TEST_SAMPLES, 0.1);
        
        ProgressBar gdProgress(MAX_EPOCHS, "Gradient Descent ", 25);
        auto gdResult = trainWithGradientDescent(trainData, testData, &gdProgress);
        gdProgress.finish("→ " + std::to_string(static_cast<int>(gdResult.finalAccuracy * 100)) + "%");
        gdMetrics.push_back(gdResult);
        
        ProgressBar n3Progress(MAX_EPOCHS, "N=3 (33% elim)   ", 25);
        auto n3Result = trainWithQuaylynPopulation(trainData, testData, POPULATION_SIZE_N3, ELIMINATION_COUNT_N3, &n3Progress);
        n3Progress.finish("→ " + std::to_string(static_cast<int>(n3Result.finalAccuracy * 100)) + "%");
        n3Metrics.push_back(n3Result);
        
        ProgressBar n9Progress(MAX_EPOCHS, "N=9 (11% elim)   ", 25);
        auto n9Result = trainWithQuaylynPopulation(trainData, testData, POPULATION_SIZE_N9, ELIMINATION_COUNT_N9, &n9Progress);
        n9Progress.finish("→ " + std::to_string(static_cast<int>(n9Result.finalAccuracy * 100)) + "%");
        n9Metrics.push_back(n9Result);
        
        ProgressBar n3HybridProgress(MAX_EPOCHS, "N=3 + Backprop   ", 25);
        auto n3HybridResult = trainHybrid(trainData, testData, POPULATION_SIZE_N3, ELIMINATION_COUNT_N3, &n3HybridProgress);
        n3HybridProgress.finish("→ " + std::to_string(static_cast<int>(n3HybridResult.finalAccuracy * 100)) + "%");
        n3HybridMetrics.push_back(n3HybridResult);
        
        ProgressBar n9HybridProgress(MAX_EPOCHS, "N=9 + Backprop   ", 25);
        auto n9HybridResult = trainHybrid(trainData, testData, POPULATION_SIZE_N9, ELIMINATION_COUNT_N9, &n9HybridProgress);
        n9HybridProgress.finish("→ " + std::to_string(static_cast<int>(n9HybridResult.finalAccuracy * 100)) + "%");
        n9HybridMetrics.push_back(n9HybridResult);
        
        std::cout << "  ╰───────────────────────────────────────────────────────────────────────────────╯\n\n";
    }
    
    // Calculate averages
    auto avgMetrics = [](const std::vector<TrainingMetrics>& metrics) {
        TrainingMetrics avg = {};
        for (const auto& m : metrics) {
            avg.trainingTime += m.trainingTime;
            avg.finalAccuracy += m.finalAccuracy;
            avg.inferenceTime += m.inferenceTime;
            avg.finalLoss += m.finalLoss;
        }
        int n = metrics.size();
        avg.trainingTime /= n;
        avg.finalAccuracy /= n;
        avg.inferenceTime /= n;
        avg.finalLoss /= n;
        return avg;
    };
    
    auto gdAvg = avgMetrics(gdMetrics);
    auto n3Avg = avgMetrics(n3Metrics);
    auto n9Avg = avgMetrics(n9Metrics);
    auto n3HybridAvg = avgMetrics(n3HybridMetrics);
    auto n9HybridAvg = avgMetrics(n9HybridMetrics);
    
    // Results
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                         COMPARISON RESULTS: POPULATION-BASED                                                            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Metric                │ Gradient   │ N=3 Elim   │ N=9 Elim   │ N=3+Backprop │ N=9+Backprop │ Winner\n";
    std::cout << "  ──────────────────────┼────────────┼────────────┼────────────┼──────────────┼──────────────┼──────────────\n";
    
    // Training time
    std::cout << "  Training Time (ms)    │ " << std::setw(10) << gdAvg.trainingTime 
              << " │ " << std::setw(10) << n3Avg.trainingTime 
              << " │ " << std::setw(10) << n9Avg.trainingTime
              << " │ " << std::setw(12) << n3HybridAvg.trainingTime
              << " │ " << std::setw(12) << n9HybridAvg.trainingTime << " │ ";
    double minTime = std::min({gdAvg.trainingTime, n3Avg.trainingTime, n9Avg.trainingTime, 
                               n3HybridAvg.trainingTime, n9HybridAvg.trainingTime});
    if (minTime == gdAvg.trainingTime) std::cout << "Gradient\n";
    else if (minTime == n3Avg.trainingTime) std::cout << "N=3\n";
    else if (minTime == n9Avg.trainingTime) std::cout << "N=9\n";
    else if (minTime == n3HybridAvg.trainingTime) std::cout << "N=3+BP\n";
    else std::cout << "N=9+BP\n";
    
    // Accuracy
    std::cout << "  Final Accuracy (%)    │ " << std::setw(9) << (gdAvg.finalAccuracy * 100) << "%" 
              << " │ " << std::setw(9) << (n3Avg.finalAccuracy * 100) << "%"
              << " │ " << std::setw(9) << (n9Avg.finalAccuracy * 100) << "%"
              << " │ " << std::setw(11) << (n3HybridAvg.finalAccuracy * 100) << "%"
              << " │ " << std::setw(11) << (n9HybridAvg.finalAccuracy * 100) << "% │ ";
    double maxAcc = std::max({gdAvg.finalAccuracy, n3Avg.finalAccuracy, n9Avg.finalAccuracy,
                              n3HybridAvg.finalAccuracy, n9HybridAvg.finalAccuracy});
    if (maxAcc == gdAvg.finalAccuracy) std::cout << "Gradient\n";
    else if (maxAcc == n3Avg.finalAccuracy) std::cout << "N=3 (+" << std::setprecision(1) << (n3Avg.finalAccuracy - gdAvg.finalAccuracy) * 100 << "%)\n";
    else if (maxAcc == n9Avg.finalAccuracy) std::cout << "N=9 (+" << std::setprecision(1) << (n9Avg.finalAccuracy - gdAvg.finalAccuracy) * 100 << "%)\n";
    else if (maxAcc == n3HybridAvg.finalAccuracy) std::cout << "N=3+BP (+" << std::setprecision(1) << (n3HybridAvg.finalAccuracy - gdAvg.finalAccuracy) * 100 << "%)\n";
    else std::cout << "N=9+BP (+" << std::setprecision(1) << (n9HybridAvg.finalAccuracy - gdAvg.finalAccuracy) * 100 << "%)\n";
    std::cout << std::setprecision(2);
    
    // Loss
    std::cout << std::setprecision(4);
    std::cout << "  Final Loss            │ " << std::setw(10) << gdAvg.finalLoss 
              << " │ " << std::setw(10) << n3Avg.finalLoss 
              << " │ " << std::setw(10) << n9Avg.finalLoss
              << " │ " << std::setw(12) << n3HybridAvg.finalLoss
              << " │ " << std::setw(12) << n9HybridAvg.finalLoss << " │ ";
    double minLoss = std::min({gdAvg.finalLoss, n3Avg.finalLoss, n9Avg.finalLoss,
                               n3HybridAvg.finalLoss, n9HybridAvg.finalLoss});
    if (minLoss == gdAvg.finalLoss) std::cout << "Gradient\n";
    else if (minLoss == n3Avg.finalLoss) std::cout << "N=3\n";
    else if (minLoss == n9Avg.finalLoss) std::cout << "N=9\n";
    else if (minLoss == n3HybridAvg.finalLoss) std::cout << "N=3+BP\n";
    else std::cout << "N=9+BP\n";
    
    // Summary
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                  KEY FINDINGS                                                                           ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "  POPULATION-BASED ELIMINATION (Quaylyn's Law):\n";
    std::cout << "  • N=3: Eliminate 33% per generation (Quaylyn's optimal)\n";
    std::cout << "  • N=9: Eliminate 11% per generation (conservative)\n";
    std::cout << "  • +Backprop: Use elimination to find good starting point, then fine-tune with gradients\n\n";
    
    double n3VsGd = (n3Avg.finalAccuracy - gdAvg.finalAccuracy) * 100;
    double n9VsGd = (n9Avg.finalAccuracy - gdAvg.finalAccuracy) * 100;
    double n3HybridVsGd = (n3HybridAvg.finalAccuracy - gdAvg.finalAccuracy) * 100;
    double n9HybridVsGd = (n9HybridAvg.finalAccuracy - gdAvg.finalAccuracy) * 100;
    
    std::cout << std::setprecision(1);
    std::cout << "  Accuracy vs Gradient Descent:\n";
    std::cout << "    N=3 (pure elim):   " << (n3VsGd >= 0 ? "+" : "") << n3VsGd << "%\n";
    std::cout << "    N=9 (pure elim):   " << (n9VsGd >= 0 ? "+" : "") << n9VsGd << "%\n";
    std::cout << "    N=3 + Backprop:    " << (n3HybridVsGd >= 0 ? "+" : "") << n3HybridVsGd << "%\n";
    std::cout << "    N=9 + Backprop:    " << (n9HybridVsGd >= 0 ? "+" : "") << n9HybridVsGd << "%\n";
    
    std::cout << "\n  HYPOTHESIS: Elimination finds better starting points than random init,\n";
    std::cout << "              then backprop can fine-tune without getting stuck in local minima.\n";
    
    std::cout << "\n";
    
    return 0;
}
