/**
 * Neural Network Training Comparison #4: WEIGHT PRUNING ELIMINATION
 * 
 * Strategy: Eliminate weakest weights instead of networks
 * - Train network, then iteratively prune smallest weights
 * - Weights with magnitude < threshold are eliminated (set to 0)
 * - Remaining weights are re-trained
 * - Combines gradient descent with Quaylyn's elimination principle
 * 
 * Advantage: Creates sparse, efficient networks
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

// Pruning settings
constexpr double PRUNE_PERCENTAGE_N3 = 0.33;  // Eliminate 33% smallest weights (N=3)
constexpr double PRUNE_PERCENTAGE_N9 = 0.11;  // Eliminate 11% smallest weights (N=9)
constexpr int PRUNE_INTERVAL = 50;            // Prune every N epochs
constexpr int RETRAIN_EPOCHS = 10;            // Retrain after pruning

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
// NEURAL NETWORK WITH PRUNING
// ============================================================================

struct NeuralNetwork {
    std::vector<std::vector<double>> weights_ih, weights_ho;
    std::vector<std::vector<bool>> mask_ih, mask_ho;  // Pruning masks
    std::vector<double> bias_h, bias_o;
    std::vector<double> hidden_raw, hidden_act, output_raw, output_act;
    
    NeuralNetwork() {
        weights_ih.resize(HIDDEN_NEURONS, std::vector<double>(INPUT_SIZE));
        weights_ho.resize(OUTPUT_SIZE, std::vector<double>(HIDDEN_NEURONS));
        mask_ih.resize(HIDDEN_NEURONS, std::vector<bool>(INPUT_SIZE, true));
        mask_ho.resize(OUTPUT_SIZE, std::vector<bool>(HIDDEN_NEURONS, true));
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
            for (int j = 0; j < INPUT_SIZE; j++) {
                if (mask_ih[i][j]) hidden_raw[i] += weights_ih[i][j] * input[j];
            }
            hidden_act[i] = sigmoid(hidden_raw[i]);
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output_raw[i] = bias_o[i];
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                if (mask_ho[i][j]) output_raw[i] += weights_ho[i][j] * hidden_act[j];
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
        for (const auto& dp : data) if (predict(dp.input) == dp.label) correct++;
        return static_cast<double>(correct) / data.size();
    }
    
    // Count active (non-pruned) weights
    int countActiveWeights() {
        int count = 0;
        for (int i = 0; i < HIDDEN_NEURONS; i++)
            for (int j = 0; j < INPUT_SIZE; j++)
                if (mask_ih[i][j]) count++;
        for (int i = 0; i < OUTPUT_SIZE; i++)
            for (int j = 0; j < HIDDEN_NEURONS; j++)
                if (mask_ho[i][j]) count++;
        return count;
    }
    
    // Prune smallest weights (Quaylyn's elimination)
    void pruneSmallestWeights(double percentage) {
        // Collect all active weight magnitudes
        std::vector<std::pair<double, std::tuple<int, int, int>>> weights;  // (|w|, layer, i, j)
        
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                if (mask_ih[i][j]) {
                    weights.push_back({std::abs(weights_ih[i][j]), {0, i, j}});
                }
            }
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                if (mask_ho[i][j]) {
                    weights.push_back({std::abs(weights_ho[i][j]), {1, i, j}});
                }
            }
        }
        
        if (weights.empty()) return;
        
        // Sort by magnitude (smallest first)
        std::sort(weights.begin(), weights.end());
        
        // Eliminate smallest 33%
        int numToPrune = static_cast<int>(weights.size() * percentage);
        numToPrune = std::min(numToPrune, (int)weights.size() - 5);  // Keep at least 5 weights
        
        for (int k = 0; k < numToPrune; k++) {
            auto [layer, i, j] = weights[k].second;
            if (layer == 0) {
                mask_ih[i][j] = false;
                weights_ih[i][j] = 0.0;
            } else {
                mask_ho[i][j] = false;
                weights_ho[i][j] = 0.0;
            }
        }
    }
};

// ============================================================================
// GRADIENT DESCENT TRAINER (standard)
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
            for (int j = 0; j < INPUT_SIZE; j++) {
                if (nn.mask_ih[i][j]) nn.weights_ih[i][j] -= scale * grad_ih[i][j];
            }
            nn.bias_h[i] -= scale * grad_bh[i];
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                if (nn.mask_ho[i][j]) nn.weights_ho[i][j] -= scale * grad_ho[i][j];
            }
            nn.bias_o[i] -= scale * grad_bo[i];
        }
    }
};

// ============================================================================
// TRAINING METRICS & RUNNERS
// ============================================================================

struct TrainingMetrics {
    double trainingTime, finalAccuracy, inferenceTime, finalLoss;
    int epochsToConverge;
    int activeWeights;
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
    metrics.activeWeights = nn.countActiveWeights();
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) nn.predict(testData[i % testData.size()].input);
    auto t2 = std::chrono::high_resolution_clock::now();
    metrics.inferenceTime = std::chrono::duration<double, std::micro>(t2 - t1).count() / 1000.0;
    
    return metrics;
}

TrainingMetrics trainWithPruning(const std::vector<DataPoint>& trainData,
                                  const std::vector<DataPoint>& testData,
                                  double prunePercentage,
                                  ProgressBar* progress = nullptr) {
    TrainingMetrics metrics;
    NeuralNetwork nn;
    GradientDescentTrainer trainer;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    metrics.epochsToConverge = MAX_EPOCHS;
    metrics.converged = false;
    
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        trainer.train(nn, trainData, LEARNING_RATE);
        
        // Prune every PRUNE_INTERVAL epochs
        if (epoch > 0 && epoch % PRUNE_INTERVAL == 0) {
            nn.pruneSmallestWeights(prunePercentage);
            
            // Retrain after pruning
            for (int r = 0; r < RETRAIN_EPOCHS; r++) {
                trainer.train(nn, trainData, LEARNING_RATE);
            }
        }
        
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
    metrics.activeWeights = nn.countActiveWeights();
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) nn.predict(testData[i % testData.size()].input);
    auto t2 = std::chrono::high_resolution_clock::now();
    metrics.inferenceTime = std::chrono::duration<double, std::micro>(t2 - t1).count() / 1000.0;
    
    return metrics;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       NEURAL NETWORK TRAINING #4: WEIGHT PRUNING ELIMINATION                               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Strategy: Eliminate weakest weights (magnitude-based pruning)\n";
    std::cout << "  • N=3: Prune 33% smallest weights every " << PRUNE_INTERVAL << " epochs\n";
    std::cout << "  • N=9: Prune 11% smallest weights every " << PRUNE_INTERVAL << " epochs\n";
    std::cout << "  • Retrain " << RETRAIN_EPOCHS << " epochs after each prune\n";
    std::cout << "  • Combines gradient descent with Quaylyn's elimination\n\n";
    
    std::vector<TrainingMetrics> gdMetrics, n3Metrics, n9Metrics;
    
    std::cout << "Running " << NUM_TRIALS << " trials...\n\n";
    
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        std::cout << "  ╭─ Trial " << (trial + 1) << "/" << NUM_TRIALS << " ───────────────────────────────────────────────────────────────╮\n";
        
        auto trainData = generateDataset(TRAINING_SAMPLES, 0.1);
        auto testData = generateDataset(TEST_SAMPLES, 0.1);
        
        ProgressBar gdProg(MAX_EPOCHS, "Gradient Descent ", 25);
        auto gdResult = trainWithGradientDescent(trainData, testData, &gdProg);
        gdProg.finish("→ " + std::to_string(static_cast<int>(gdResult.finalAccuracy * 100)) + "% (" + 
                      std::to_string(gdResult.activeWeights) + " wts)");
        gdMetrics.push_back(gdResult);
        
        ProgressBar n3Prog(MAX_EPOCHS, "N=3 (33% prune)  ", 25);
        auto n3Result = trainWithPruning(trainData, testData, PRUNE_PERCENTAGE_N3, &n3Prog);
        n3Prog.finish("→ " + std::to_string(static_cast<int>(n3Result.finalAccuracy * 100)) + "% (" + 
                      std::to_string(n3Result.activeWeights) + " wts)");
        n3Metrics.push_back(n3Result);
        
        ProgressBar n9Prog(MAX_EPOCHS, "N=9 (11% prune)  ", 25);
        auto n9Result = trainWithPruning(trainData, testData, PRUNE_PERCENTAGE_N9, &n9Prog);
        n9Prog.finish("→ " + std::to_string(static_cast<int>(n9Result.finalAccuracy * 100)) + "% (" + 
                      std::to_string(n9Result.activeWeights) + " wts)");
        n9Metrics.push_back(n9Result);
        
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
            a.activeWeights += x.activeWeights;
        }
        int n = m.size();
        a.trainingTime /= n; a.finalAccuracy /= n; a.finalLoss /= n; 
        a.inferenceTime /= n; a.activeWeights /= n;
        return a;
    };
    
    auto gdAvg = avg(gdMetrics), n3Avg = avg(n3Metrics), n9Avg = avg(n9Metrics);
    
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                              RESULTS: WEIGHT PRUNING                                                   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Metric                    │ Gradient Descent │ N=3 (33% prune)  │ N=9 (11% prune)  │ Winner\n";
    std::cout << "  ──────────────────────────┼──────────────────┼──────────────────┼──────────────────┼────────────\n";
    
    std::cout << "  Training Time (ms)        │ " << std::setw(16) << gdAvg.trainingTime 
              << " │ " << std::setw(16) << n3Avg.trainingTime 
              << " │ " << std::setw(16) << n9Avg.trainingTime << " │ ";
    if (gdAvg.trainingTime <= n3Avg.trainingTime && gdAvg.trainingTime <= n9Avg.trainingTime)
        std::cout << "Gradient\n";
    else if (n3Avg.trainingTime <= n9Avg.trainingTime)
        std::cout << "N=3\n";
    else
        std::cout << "N=9\n";
    
    std::cout << "  Final Accuracy (%)        │ " << std::setw(15) << (gdAvg.finalAccuracy * 100) << "%" 
              << " │ " << std::setw(15) << (n3Avg.finalAccuracy * 100) << "%"
              << " │ " << std::setw(15) << (n9Avg.finalAccuracy * 100) << "%" << " │ ";
    if (gdAvg.finalAccuracy >= n3Avg.finalAccuracy && gdAvg.finalAccuracy >= n9Avg.finalAccuracy)
        std::cout << "Gradient\n";
    else if (n3Avg.finalAccuracy >= n9Avg.finalAccuracy)
        std::cout << "N=3 (+" << std::setprecision(1) << (n3Avg.finalAccuracy - gdAvg.finalAccuracy) * 100 << "%)\n";
    else
        std::cout << "N=9 (+" << std::setprecision(1) << (n9Avg.finalAccuracy - gdAvg.finalAccuracy) * 100 << "%)\n";
    
    std::cout << std::setprecision(4);
    std::cout << "  Final Loss                │ " << std::setw(16) << gdAvg.finalLoss 
              << " │ " << std::setw(16) << n3Avg.finalLoss 
              << " │ " << std::setw(16) << n9Avg.finalLoss << " │ ";
    if (gdAvg.finalLoss <= n3Avg.finalLoss && gdAvg.finalLoss <= n9Avg.finalLoss)
        std::cout << "Gradient\n";
    else if (n3Avg.finalLoss <= n9Avg.finalLoss)
        std::cout << "N=3\n";
    else
        std::cout << "N=9\n";
    
    std::cout << std::setprecision(0);
    std::cout << "  Active Weights            │ " << std::setw(16) << gdAvg.activeWeights 
              << " │ " << std::setw(16) << n3Avg.activeWeights 
              << " │ " << std::setw(16) << n9Avg.activeWeights << " │ Smaller=Better\n";
    
    std::cout << "\n  WEIGHT PRUNING PRINCIPLE:\n";
    std::cout << "  • N=3: Aggressive pruning (33% per cycle) = smallest model\n";
    std::cout << "  • N=9: Conservative pruning (11% per cycle) = retain more weights\n";
    std::cout << "  • Creates sparse network = faster inference\n";
    
    double n3VsGd = (n3Avg.finalAccuracy - gdAvg.finalAccuracy) * 100;
    double n9VsGd = (n9Avg.finalAccuracy - gdAvg.finalAccuracy) * 100;
    
    std::cout << std::setprecision(1);
    std::cout << "\n  N=3 vs Gradient: " << (n3VsGd >= 0 ? "+" : "") << n3VsGd << "% accuracy, ";
    std::cout << n3Avg.activeWeights << "/" << gdAvg.activeWeights << " weights\n";
    std::cout << "  N=9 vs Gradient: " << (n9VsGd >= 0 ? "+" : "") << n9VsGd << "% accuracy, ";
    std::cout << n9Avg.activeWeights << "/" << gdAvg.activeWeights << " weights\n";
    
    std::cout << "\n";
    return 0;
}
