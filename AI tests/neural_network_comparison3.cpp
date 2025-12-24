/**
 * Neural Network Training Comparison #3: TOURNAMENT ELIMINATION
 * 
 * Strategy: Networks compete head-to-head in tournaments
 * - Pairs of networks compete on same data batch
 * - Loser is eliminated, winner advances
 * - Losers replaced by mutated winners
 * - True to Quaylyn's Law: only eliminate losers, never claim "best"
 * 
 * Advantage: Direct comparison reduces noise in evaluation
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

constexpr int HIDDEN_NEURONS = 8;
constexpr int INPUT_SIZE = 4;
constexpr int OUTPUT_SIZE = 2;
constexpr int TRAINING_SAMPLES = 1000;
constexpr int TEST_SAMPLES = 200;
constexpr int MAX_EPOCHS = 500;
constexpr double LEARNING_RATE = 0.1;
constexpr double TARGET_ACCURACY = 0.95;
constexpr int NUM_TRIALS = 10;

// Tournament settings
// N=3: Tournament of 3, eliminate 1 (33%)
constexpr int POPULATION_SIZE = 9;
constexpr int TOURNAMENT_SIZE_N3 = 3;      // Compare 3 at a time, eliminate 1 (33%)
constexpr int TOURNAMENT_SIZE_N9 = 9;      // Compare 9 at a time, eliminate 1 (11%)
constexpr double MUTATION_RATE = 0.3;
constexpr double MUTATION_STRENGTH = 0.5;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

class ProgressBar {
private:
    int total, current, barWidth;
    std::string prefix;
    std::chrono::steady_clock::time_point startTime;
public:
    ProgressBar(int t, const std::string& p = "", int w = 25) 
        : total(t), current(0), barWidth(w), prefix(p) {
        startTime = std::chrono::steady_clock::now();
    }
    void update(int v) { current = v; display(); }
    void display() {
        float progress = static_cast<float>(current) / total;
        int filled = static_cast<int>(barWidth * progress);
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - startTime).count();
        std::cout << "\r  " << prefix << " [";
        for (int i = 0; i < barWidth; i++) {
            if (i < filled) std::cout << "█";
            else if (i == filled) std::cout << "▓";
            else std::cout << "░";
        }
        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100) << "% "
                  << current << "/" << total << " [" << std::fixed << std::setprecision(1) 
                  << elapsed << "s]    " << std::flush;
    }
    void finish(const std::string& msg = "") { current = total; display(); std::cout << " " << msg << "\n"; }
};

double randomDouble(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-std::clamp(x, -500.0, 500.0))); }
double sigmoidDerivative(double x) { double s = sigmoid(x); return s * (1.0 - s); }

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
        
        for (int j = 0; j < INPUT_SIZE; j++) dp.input[j] = randomDouble(0.0, 1.0);
        
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
    std::vector<std::vector<double>> weights_ih, weights_ho;
    std::vector<double> bias_h, bias_o;
    std::vector<double> hidden_raw, hidden_act, output_raw, output_act;
    
    NeuralNetwork() {
        weights_ih.resize(HIDDEN_NEURONS, std::vector<double>(INPUT_SIZE));
        weights_ho.resize(OUTPUT_SIZE, std::vector<double>(HIDDEN_NEURONS));
        bias_h.resize(HIDDEN_NEURONS); bias_o.resize(OUTPUT_SIZE);
        hidden_raw.resize(HIDDEN_NEURONS); hidden_act.resize(HIDDEN_NEURONS);
        output_raw.resize(OUTPUT_SIZE); output_act.resize(OUTPUT_SIZE);
        randomizeWeights();
    }
    
    void randomizeWeights() {
        double range = std::sqrt(6.0 / (INPUT_SIZE + HIDDEN_NEURONS));
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) weights_ih[i][j] = randomDouble(-range, range);
            bias_h[i] = randomDouble(-0.1, 0.1);
        }
        range = std::sqrt(6.0 / (HIDDEN_NEURONS + OUTPUT_SIZE));
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) weights_ho[i][j] = randomDouble(-range, range);
            bias_o[i] = randomDouble(-0.1, 0.1);
        }
    }
    
    std::vector<double> forward(const std::vector<double>& input) {
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            hidden_raw[i] = bias_h[i];
            for (int j = 0; j < INPUT_SIZE; j++) hidden_raw[i] += weights_ih[i][j] * input[j];
            hidden_act[i] = sigmoid(hidden_raw[i]);
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output_raw[i] = bias_o[i];
            for (int j = 0; j < HIDDEN_NEURONS; j++) output_raw[i] += weights_ho[i][j] * hidden_act[j];
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
        for (const auto& dp : data) if (predict(dp.input) == dp.label) correct++;
        return static_cast<double>(correct) / data.size();
    }
    
    void copyFrom(const NeuralNetwork& other) {
        weights_ih = other.weights_ih; weights_ho = other.weights_ho;
        bias_h = other.bias_h; bias_o = other.bias_o;
    }
    
    void mutate(double rate, double strength) {
        std::normal_distribution<double> noise(0.0, strength);
        std::uniform_real_distribution<double> chance(0.0, 1.0);
        for (auto& row : weights_ih) for (auto& w : row) if (chance(rng) < rate) w += noise(rng);
        for (auto& b : bias_h) if (chance(rng) < rate) b += noise(rng);
        for (auto& row : weights_ho) for (auto& w : row) if (chance(rng) < rate) w += noise(rng);
        for (auto& b : bias_o) if (chance(rng) < rate) b += noise(rng);
    }
    
    void crossover(const NeuralNetwork& p1, const NeuralNetwork& p2) {
        std::uniform_real_distribution<double> blend(0.0, 1.0);
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                double t = blend(rng);
                weights_ih[i][j] = t * p1.weights_ih[i][j] + (1-t) * p2.weights_ih[i][j];
            }
            double t = blend(rng);
            bias_h[i] = t * p1.bias_h[i] + (1-t) * p2.bias_h[i];
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                double t = blend(rng);
                weights_ho[i][j] = t * p1.weights_ho[i][j] + (1-t) * p2.weights_ho[i][j];
            }
            double t = blend(rng);
            bias_o[i] = t * p1.bias_o[i] + (1-t) * p2.bias_o[i];
        }
    }
};

// ============================================================================
// GRADIENT DESCENT TRAINER
// ============================================================================

struct GradientDescentTrainer {
    void train(NeuralNetwork& nn, const std::vector<DataPoint>& data, double lr) {
        std::vector<std::vector<double>> grad_ih(HIDDEN_NEURONS, std::vector<double>(INPUT_SIZE, 0.0));
        std::vector<std::vector<double>> grad_ho(OUTPUT_SIZE, std::vector<double>(HIDDEN_NEURONS, 0.0));
        std::vector<double> grad_bh(HIDDEN_NEURONS, 0.0), grad_bo(OUTPUT_SIZE, 0.0);
        
        for (const auto& dp : data) {
            nn.forward(dp.input);
            std::vector<double> output_delta(OUTPUT_SIZE);
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                output_delta[i] = (nn.output_act[i] - dp.target[i]) * sigmoidDerivative(nn.output_raw[i]);
            }
            std::vector<double> hidden_delta(HIDDEN_NEURONS);
            for (int i = 0; i < HIDDEN_NEURONS; i++) {
                double error = 0.0;
                for (int j = 0; j < OUTPUT_SIZE; j++) error += output_delta[j] * nn.weights_ho[j][i];
                hidden_delta[i] = error * sigmoidDerivative(nn.hidden_raw[i]);
            }
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                for (int j = 0; j < HIDDEN_NEURONS; j++) grad_ho[i][j] += output_delta[i] * nn.hidden_act[j];
                grad_bo[i] += output_delta[i];
            }
            for (int i = 0; i < HIDDEN_NEURONS; i++) {
                for (int j = 0; j < INPUT_SIZE; j++) grad_ih[i][j] += hidden_delta[i] * dp.input[j];
                grad_bh[i] += hidden_delta[i];
            }
        }
        
        double scale = lr / data.size();
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) nn.weights_ih[i][j] -= scale * grad_ih[i][j];
            nn.bias_h[i] -= scale * grad_bh[i];
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) nn.weights_ho[i][j] -= scale * grad_ho[i][j];
            nn.bias_o[i] -= scale * grad_bo[i];
        }
    }
};

// ============================================================================
// TOURNAMENT ELIMINATION TRAINER
// ============================================================================

struct TournamentEliminationTrainer {
    int tournamentSize;
    
    TournamentEliminationTrainer(int tSize) : tournamentSize(tSize) {}
    
    NeuralNetwork train(const std::vector<DataPoint>& valData, int maxGen, ProgressBar* prog = nullptr) {
        std::vector<NeuralNetwork> population(POPULATION_SIZE);
        NeuralNetwork bestNetwork;
        double bestLoss = 1e9;
        
        for (int gen = 0; gen < maxGen; gen++) {
            // Run multiple tournaments per generation
            int numTournaments = POPULATION_SIZE / tournamentSize;
            
            for (int t = 0; t < numTournaments; t++) {
                // Select random participants for tournament
                std::vector<int> participants;
                std::vector<int> available;
                for (int i = 0; i < POPULATION_SIZE; i++) available.push_back(i);
                std::shuffle(available.begin(), available.end(), rng);
                
                for (int i = 0; i < tournamentSize && i < (int)available.size(); i++) {
                    participants.push_back(available[i]);
                }
                
                // Evaluate participants on SAME data batch (fair comparison)
                std::vector<DataPoint> batch(valData.begin(), 
                    valData.begin() + std::min(50, (int)valData.size()));
                
                std::vector<std::pair<double, int>> scores;
                for (int idx : participants) {
                    double loss = population[idx].calculateLoss(batch);
                    scores.push_back({loss, idx});
                    
                    if (loss < bestLoss) {
                        bestLoss = loss;
                        bestNetwork.copyFrom(population[idx]);
                    }
                }
                
                // Sort: worst (highest loss) first
                std::sort(scores.begin(), scores.end(), 
                    [](const auto& a, const auto& b) { return a.first > b.first; });
                
                // Eliminate worst 1 from tournament (33% of 3)
                int loserIdx = scores[0].second;
                int winnerIdx = scores.back().second;
                
                // Replace loser with mutated winner
                population[loserIdx].copyFrom(population[winnerIdx]);
                population[loserIdx].mutate(MUTATION_RATE, MUTATION_STRENGTH);
            }
            
            if (prog && gen % 5 == 0) prog->update(gen);
        }
        
        if (prog) prog->update(maxGen);
        return bestNetwork;
    }
};

// ============================================================================
// TRAINING METRICS & RUNNERS
// ============================================================================

struct TrainingMetrics {
    double trainingTime, finalAccuracy, inferenceTime, finalLoss;
    int epochsToConverge;
    bool converged;
};

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
        double acc = nn.calculateAccuracy(testData);
        if (acc >= TARGET_ACCURACY && !metrics.converged) {
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
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) nn.predict(testData[i % testData.size()].input);
    auto t2 = std::chrono::high_resolution_clock::now();
    metrics.inferenceTime = std::chrono::duration<double, std::micro>(t2 - t1).count() / 1000.0;
    
    return metrics;
}

TrainingMetrics trainWithTournament(const std::vector<DataPoint>& trainData,
                                     const std::vector<DataPoint>& testData,
                                     int tournamentSize,
                                     ProgressBar* progress = nullptr) {
    TrainingMetrics metrics;
    TournamentEliminationTrainer trainer(tournamentSize);
    
    std::vector<DataPoint> valData(trainData.begin(), trainData.begin() + trainData.size() / 5);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    NeuralNetwork bestNet = trainer.train(valData, MAX_EPOCHS, progress);
    auto endTime = std::chrono::high_resolution_clock::now();
    
    metrics.trainingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    metrics.finalAccuracy = bestNet.calculateAccuracy(testData);
    metrics.finalLoss = bestNet.calculateLoss(testData);
    metrics.converged = metrics.finalAccuracy >= TARGET_ACCURACY;
    metrics.epochsToConverge = MAX_EPOCHS;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) bestNet.predict(testData[i % testData.size()].input);
    auto t2 = std::chrono::high_resolution_clock::now();
    metrics.inferenceTime = std::chrono::duration<double, std::micro>(t2 - t1).count() / 1000.0;
    
    return metrics;
}

// HYBRID: Tournament elimination first, then backprop to fine-tune
TrainingMetrics trainHybrid(const std::vector<DataPoint>& trainData,
                            const std::vector<DataPoint>& testData,
                            int tournamentSize,
                            ProgressBar* progress = nullptr) {
    TrainingMetrics metrics;
    TournamentEliminationTrainer tourneyTrainer(tournamentSize);
    GradientDescentTrainer gdTrainer;
    
    std::vector<DataPoint> valData(trainData.begin(), trainData.begin() + trainData.size() / 5);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Phase 1: Tournament elimination (half epochs)
    int elimEpochs = MAX_EPOCHS / 2;
    NeuralNetwork bestNet = tourneyTrainer.train(valData, elimEpochs, nullptr);
    
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
    metrics.converged = metrics.finalAccuracy >= TARGET_ACCURACY;
    metrics.epochsToConverge = MAX_EPOCHS;
    
    auto t1h = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) bestNet.predict(testData[i % testData.size()].input);
    auto t2h = std::chrono::high_resolution_clock::now();
    metrics.inferenceTime = std::chrono::duration<double, std::micro>(t2h - t1h).count() / 1000.0;
    
    return metrics;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       NEURAL NETWORK TRAINING #3: TOURNAMENT ELIMINATION                                   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Strategy: Networks compete head-to-head, losers eliminated\n";
    std::cout << "  • Population: " << POPULATION_SIZE << " networks\n";
    std::cout << "  • N=3: Tournament size 3, eliminate 1 (33%)\n";
    std::cout << "  • N=9: Tournament size 9, eliminate 1 (11%)\n";
    std::cout << "  • Direct comparison on same data batch = fair evaluation\n\n";
    
    std::vector<TrainingMetrics> gdMetrics, n3Metrics, n9Metrics, n3HybridMetrics, n9HybridMetrics;
    
    std::cout << "Running " << NUM_TRIALS << " trials...\n\n";
    
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        std::cout << "  ╭─ Trial " << (trial + 1) << "/" << NUM_TRIALS << " ───────────────────────────────────────────────────────────────╮\n";
        
        auto trainData = generateDataset(TRAINING_SAMPLES, 0.1);
        auto testData = generateDataset(TEST_SAMPLES, 0.1);
        
        ProgressBar gdProg(MAX_EPOCHS, "Gradient Descent ", 25);
        auto gdResult = trainWithGradientDescent(trainData, testData, &gdProg);
        gdProg.finish("→ " + std::to_string(static_cast<int>(gdResult.finalAccuracy * 100)) + "%");
        gdMetrics.push_back(gdResult);
        
        ProgressBar n3Prog(MAX_EPOCHS, "N=3 (33% elim)   ", 25);
        auto n3Result = trainWithTournament(trainData, testData, TOURNAMENT_SIZE_N3, &n3Prog);
        n3Prog.finish("→ " + std::to_string(static_cast<int>(n3Result.finalAccuracy * 100)) + "%");
        n3Metrics.push_back(n3Result);
        
        ProgressBar n9Prog(MAX_EPOCHS, "N=9 (11% elim)   ", 25);
        auto n9Result = trainWithTournament(trainData, testData, TOURNAMENT_SIZE_N9, &n9Prog);
        n9Prog.finish("→ " + std::to_string(static_cast<int>(n9Result.finalAccuracy * 100)) + "%");
        n9Metrics.push_back(n9Result);
        
        ProgressBar n3HybridProg(MAX_EPOCHS, "N=3 + Backprop   ", 25);
        auto n3HybridResult = trainHybrid(trainData, testData, TOURNAMENT_SIZE_N3, &n3HybridProg);
        n3HybridProg.finish("→ " + std::to_string(static_cast<int>(n3HybridResult.finalAccuracy * 100)) + "%");
        n3HybridMetrics.push_back(n3HybridResult);
        
        ProgressBar n9HybridProg(MAX_EPOCHS, "N=9 + Backprop   ", 25);
        auto n9HybridResult = trainHybrid(trainData, testData, TOURNAMENT_SIZE_N9, &n9HybridProg);
        n9HybridProg.finish("→ " + std::to_string(static_cast<int>(n9HybridResult.finalAccuracy * 100)) + "%");
        n9HybridMetrics.push_back(n9HybridResult);
        
        std::cout << "  ╰───────────────────────────────────────────────────────────────────────────────╯\n\n";
    }
    
    // Averages
    auto avg = [](const std::vector<TrainingMetrics>& m) {
        TrainingMetrics a = {};
        for (const auto& x : m) {
            a.trainingTime += x.trainingTime;
            a.finalAccuracy += x.finalAccuracy;
            a.finalLoss += x.finalLoss;
            a.inferenceTime += x.inferenceTime;
        }
        int n = m.size();
        a.trainingTime /= n; a.finalAccuracy /= n; a.finalLoss /= n; a.inferenceTime /= n;
        return a;
    };
    
    auto gdAvg = avg(gdMetrics), n3Avg = avg(n3Metrics), n9Avg = avg(n9Metrics);
    auto n3HybridAvg = avg(n3HybridMetrics), n9HybridAvg = avg(n9HybridMetrics);
    
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                            RESULTS: TOURNAMENT ELIMINATION                                                                   ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Metric                    │ Gradient Descent │ N=3 (33% elim)   │ N=9 (11% elim)   │ N=3 + Backprop   │ N=9 + Backprop   │ Best\n";
    std::cout << "  ──────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼────────────────\n";
    
    // Training Time
    std::cout << "  Training Time (ms)        │ " << std::setw(16) << gdAvg.trainingTime 
              << " │ " << std::setw(16) << n3Avg.trainingTime 
              << " │ " << std::setw(16) << n9Avg.trainingTime 
              << " │ " << std::setw(16) << n3HybridAvg.trainingTime 
              << " │ " << std::setw(16) << n9HybridAvg.trainingTime << " │ ";
    std::vector<double> times = {gdAvg.trainingTime, n3Avg.trainingTime, n9Avg.trainingTime, n3HybridAvg.trainingTime, n9HybridAvg.trainingTime};
    std::vector<std::string> names = {"Gradient", "N=3", "N=9", "N=3+BP", "N=9+BP"};
    auto minTimeIt = std::min_element(times.begin(), times.end());
    std::cout << names[std::distance(times.begin(), minTimeIt)] << "\n";
    
    // Final Accuracy
    std::cout << "  Final Accuracy (%)        │ " << std::setw(15) << (gdAvg.finalAccuracy * 100) << "%" 
              << " │ " << std::setw(15) << (n3Avg.finalAccuracy * 100) << "%"
              << " │ " << std::setw(15) << (n9Avg.finalAccuracy * 100) << "%" 
              << " │ " << std::setw(15) << (n3HybridAvg.finalAccuracy * 100) << "%" 
              << " │ " << std::setw(15) << (n9HybridAvg.finalAccuracy * 100) << "%" << " │ ";
    std::vector<double> accs = {gdAvg.finalAccuracy, n3Avg.finalAccuracy, n9Avg.finalAccuracy, n3HybridAvg.finalAccuracy, n9HybridAvg.finalAccuracy};
    auto maxAccIt = std::max_element(accs.begin(), accs.end());
    std::cout << names[std::distance(accs.begin(), maxAccIt)] << "\n";
    
    // Final Loss
    std::cout << std::setprecision(4);
    std::cout << "  Final Loss                │ " << std::setw(16) << gdAvg.finalLoss 
              << " │ " << std::setw(16) << n3Avg.finalLoss 
              << " │ " << std::setw(16) << n9Avg.finalLoss 
              << " │ " << std::setw(16) << n3HybridAvg.finalLoss 
              << " │ " << std::setw(16) << n9HybridAvg.finalLoss << " │ ";
    std::vector<double> losses = {gdAvg.finalLoss, n3Avg.finalLoss, n9Avg.finalLoss, n3HybridAvg.finalLoss, n9HybridAvg.finalLoss};
    auto minLossIt = std::min_element(losses.begin(), losses.end());
    std::cout << names[std::distance(losses.begin(), minLossIt)] << "\n";

    std::cout << "\n  HYBRID APPROACH TEST:\n";
    std::cout << "  • Elimination first (50% epochs): Explores solution space, avoids local minima\n";
    std::cout << "  • Backprop second (50% epochs): Fine-tunes the best found solution\n";
    
    std::cout << std::setprecision(1);
    std::cout << "\n  ACCURACY COMPARISON vs Gradient Descent:\n";
    std::cout << "  ─────────────────────────────────────────\n";
    std::cout << "  N=3 (elimination only):  " << ((n3Avg.finalAccuracy - gdAvg.finalAccuracy) >= 0 ? "+" : "") 
              << (n3Avg.finalAccuracy - gdAvg.finalAccuracy) * 100 << "%\n";
    std::cout << "  N=9 (elimination only):  " << ((n9Avg.finalAccuracy - gdAvg.finalAccuracy) >= 0 ? "+" : "") 
              << (n9Avg.finalAccuracy - gdAvg.finalAccuracy) * 100 << "%\n";
    std::cout << "  N=3 + Backprop (hybrid): " << ((n3HybridAvg.finalAccuracy - gdAvg.finalAccuracy) >= 0 ? "+" : "") 
              << (n3HybridAvg.finalAccuracy - gdAvg.finalAccuracy) * 100 << "%\n";
    std::cout << "  N=9 + Backprop (hybrid): " << ((n9HybridAvg.finalAccuracy - gdAvg.finalAccuracy) >= 0 ? "+" : "") 
              << (n9HybridAvg.finalAccuracy - gdAvg.finalAccuracy) * 100 << "%\n";
    
    std::cout << "\n";
    return 0;
}
