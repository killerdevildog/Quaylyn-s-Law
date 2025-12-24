/**
 * Neural Network Training Comparison #8: ACTIVATION SPARSITY ELIMINATION
 * 
 * Strategy: Eliminate neurons that rarely activate
 *   - Track average activation of each hidden neuron
 *   - Low activation = neuron contributes little to output
 *   - N=3: Eliminate 33% least active neurons (Quaylyn's optimal)
 *   - N=9: Eliminate 11% least active neurons (conservative)
 * 
 * This tests Quaylyn's Law on ACTIVITY-BASED elimination
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

constexpr int INITIAL_HIDDEN = 24;          // Start with more neurons
constexpr int MIN_HIDDEN = 4;               // Minimum neurons to keep
constexpr int INPUT_SIZE = 4;
constexpr int OUTPUT_SIZE = 2;
constexpr int TRAINING_SAMPLES = 1000;
constexpr int TEST_SAMPLES = 200;
constexpr int MAX_EPOCHS = 500;
constexpr double LEARNING_RATE = 0.1;
constexpr int NUM_TRIALS = 10;

// Quaylyn's Law configurations
constexpr double ELIMINATION_RATE_N3 = 0.33;  // 33% elimination (optimal)
constexpr double ELIMINATION_RATE_N9 = 0.11;  // 11% elimination (conservative)
constexpr int ELIMINATION_INTERVAL = 100;      // Eliminate every N epochs

// ============================================================================
// UTILITIES
// ============================================================================

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

class ProgressBar {
private:
    int total, current, barWidth;
    std::string prefix;
    std::chrono::steady_clock::time_point startTime;
public:
    ProgressBar(int total, const std::string& prefix = "", int width = 25) 
        : total(total), current(0), barWidth(width), prefix(prefix) {
        startTime = std::chrono::steady_clock::now();
    }
    
    void update(int value) {
        current = value;
        float progress = static_cast<float>(current) / total;
        int filled = static_cast<int>(barWidth * progress);
        auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - startTime).count();
        
        std::cout << "\r  " << prefix << " [";
        for (int i = 0; i < barWidth; i++) {
            if (i < filled) std::cout << "█";
            else std::cout << "░";
        }
        std::cout << "] " << std::setw(3) << int(progress * 100) << "% "
                  << current << "/" << total << " [" << std::fixed 
                  << std::setprecision(1) << elapsed << "s]    " << std::flush;
    }
    
    void finish(const std::string& msg) {
        current = total;
        update(total);
        std::cout << " " << msg << "\n";
    }
};

double randomDouble(double min, double max) {
    return std::uniform_real_distribution<double>(min, max)(rng);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-std::clamp(x, -500.0, 500.0)));
}

double sigmoidDeriv(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// ============================================================================
// DATA
// ============================================================================

struct DataPoint {
    std::vector<double> input;
    std::vector<double> target;
    int label;
};

std::vector<DataPoint> generateDataset(int n, double noise = 0.1) {
    std::vector<DataPoint> data;
    std::normal_distribution<double> noiseDist(0.0, noise);
    
    for (int i = 0; i < n; i++) {
        DataPoint dp;
        dp.input.resize(INPUT_SIZE);
        dp.target.resize(OUTPUT_SIZE, 0.0);
        
        for (int j = 0; j < INPUT_SIZE; j++) {
            dp.input[j] = randomDouble(0.0, 1.0);
        }
        
        bool xor_result = (dp.input[0] > 0.5) != (dp.input[1] > 0.5);
        bool comp_result = dp.input[2] > dp.input[3];
        dp.label = (xor_result && comp_result) ? 1 : 0;
        dp.target[dp.label] = 1.0;
        
        for (int j = 0; j < INPUT_SIZE; j++) {
            dp.input[j] = std::clamp(dp.input[j] + noiseDist(rng), 0.0, 1.0);
        }
        data.push_back(dp);
    }
    return data;
}

// ============================================================================
// DYNAMIC NEURAL NETWORK (variable hidden layer size)
// ============================================================================

struct DynamicNeuralNetwork {
    int hiddenSize;
    std::vector<std::vector<double>> weights_ih;
    std::vector<std::vector<double>> weights_ho;
    std::vector<double> bias_h, bias_o;
    std::vector<double> hidden_raw, hidden_act, output_raw, output_act;
    
    // Activation tracking
    std::vector<double> activationSum;
    std::vector<int> activationCount;
    std::vector<bool> neuronActive;  // Which neurons are still active
    
    DynamicNeuralNetwork(int hidden = INITIAL_HIDDEN) : hiddenSize(hidden) {
        weights_ih.resize(hiddenSize, std::vector<double>(INPUT_SIZE));
        weights_ho.resize(OUTPUT_SIZE, std::vector<double>(hiddenSize));
        bias_h.resize(hiddenSize);
        bias_o.resize(OUTPUT_SIZE);
        hidden_raw.resize(hiddenSize);
        hidden_act.resize(hiddenSize);
        output_raw.resize(OUTPUT_SIZE);
        output_act.resize(OUTPUT_SIZE);
        
        activationSum.resize(hiddenSize, 0.0);
        activationCount.resize(hiddenSize, 0);
        neuronActive.resize(hiddenSize, true);
        
        randomizeWeights();
    }
    
    void randomizeWeights() {
        double range = std::sqrt(6.0 / (INPUT_SIZE + hiddenSize));
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                weights_ih[i][j] = randomDouble(-range, range);
            }
            bias_h[i] = randomDouble(-0.1, 0.1);
        }
        range = std::sqrt(6.0 / (hiddenSize + OUTPUT_SIZE));
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weights_ho[i][j] = randomDouble(-range, range);
            }
            bias_o[i] = randomDouble(-0.1, 0.1);
        }
    }
    
    void resetActivationTracking() {
        std::fill(activationSum.begin(), activationSum.end(), 0.0);
        std::fill(activationCount.begin(), activationCount.end(), 0);
    }
    
    std::vector<double> forward(const std::vector<double>& input, bool trackActivations = false) {
        // Hidden layer
        for (int i = 0; i < hiddenSize; i++) {
            if (!neuronActive[i]) {
                hidden_act[i] = 0.0;
                continue;
            }
            hidden_raw[i] = bias_h[i];
            for (int j = 0; j < INPUT_SIZE; j++) {
                hidden_raw[i] += weights_ih[i][j] * input[j];
            }
            hidden_act[i] = sigmoid(hidden_raw[i]);
            
            if (trackActivations) {
                activationSum[i] += hidden_act[i];
                activationCount[i]++;
            }
        }
        
        // Output layer
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output_raw[i] = bias_o[i];
            for (int j = 0; j < hiddenSize; j++) {
                if (neuronActive[j]) {
                    output_raw[i] += weights_ho[i][j] * hidden_act[j];
                }
            }
            output_act[i] = sigmoid(output_raw[i]);
        }
        return output_act;
    }
    
    int predict(const std::vector<double>& input) {
        forward(input);
        return (output_act[0] > output_act[1]) ? 0 : 1;
    }
    
    double calculateLoss(const std::vector<DataPoint>& data) {
        double loss = 0.0;
        for (const auto& dp : data) {
            forward(dp.input);
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                double diff = output_act[i] - dp.target[i];
                loss += diff * diff;
            }
        }
        return loss / data.size();
    }
    
    double calculateAccuracy(const std::vector<DataPoint>& data) {
        int correct = 0;
        for (const auto& dp : data) {
            if (predict(dp.input) == dp.label) correct++;
        }
        return static_cast<double>(correct) / data.size();
    }
    
    int countActiveNeurons() {
        int count = 0;
        for (int i = 0; i < hiddenSize; i++) {
            if (neuronActive[i]) count++;
        }
        return count;
    }
    
    std::vector<double> getAverageActivations() {
        std::vector<double> avgAct(hiddenSize, 0.0);
        for (int i = 0; i < hiddenSize; i++) {
            if (activationCount[i] > 0) {
                avgAct[i] = activationSum[i] / activationCount[i];
            }
        }
        return avgAct;
    }
};

// ============================================================================
// GRADIENT DESCENT TRAINER (for fixed 8-neuron baseline)
// ============================================================================

struct GradientDescentTrainer {
    void train(DynamicNeuralNetwork& nn, const std::vector<DataPoint>& data, double lr) {
        int H = nn.hiddenSize;
        std::vector<std::vector<double>> grad_ih(H, std::vector<double>(INPUT_SIZE, 0.0));
        std::vector<std::vector<double>> grad_ho(OUTPUT_SIZE, std::vector<double>(H, 0.0));
        std::vector<double> grad_bh(H, 0.0);
        std::vector<double> grad_bo(OUTPUT_SIZE, 0.0);
        
        for (const auto& dp : data) {
            nn.forward(dp.input);
            
            std::vector<double> output_delta(OUTPUT_SIZE);
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                double error = nn.output_act[i] - dp.target[i];
                output_delta[i] = error * sigmoidDeriv(nn.output_raw[i]);
            }
            
            std::vector<double> hidden_delta(H);
            for (int i = 0; i < H; i++) {
                if (!nn.neuronActive[i]) continue;
                double error = 0.0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    error += output_delta[j] * nn.weights_ho[j][i];
                }
                hidden_delta[i] = error * sigmoidDeriv(nn.hidden_raw[i]);
            }
            
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                for (int j = 0; j < H; j++) {
                    if (nn.neuronActive[j]) {
                        grad_ho[i][j] += output_delta[i] * nn.hidden_act[j];
                    }
                }
                grad_bo[i] += output_delta[i];
            }
            
            for (int i = 0; i < H; i++) {
                if (!nn.neuronActive[i]) continue;
                for (int j = 0; j < INPUT_SIZE; j++) {
                    grad_ih[i][j] += hidden_delta[i] * dp.input[j];
                }
                grad_bh[i] += hidden_delta[i];
            }
        }
        
        double scale = lr / data.size();
        for (int i = 0; i < H; i++) {
            if (!nn.neuronActive[i]) continue;
            for (int j = 0; j < INPUT_SIZE; j++) {
                nn.weights_ih[i][j] -= scale * grad_ih[i][j];
            }
            nn.bias_h[i] -= scale * grad_bh[i];
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < H; j++) {
                if (nn.neuronActive[j]) {
                    nn.weights_ho[i][j] -= scale * grad_ho[i][j];
                }
            }
            nn.bias_o[i] -= scale * grad_bo[i];
        }
    }
};

// ============================================================================
// ACTIVATION SPARSITY ELIMINATION TRAINER
// ============================================================================

struct ActivationSparsityTrainer {
    double eliminationRate;
    GradientDescentTrainer gdTrainer;
    
    ActivationSparsityTrainer(double elimRate) : eliminationRate(elimRate) {}
    
    void eliminateLeastActiveNeurons(DynamicNeuralNetwork& nn) {
        int activeCount = nn.countActiveNeurons();
        if (activeCount <= MIN_HIDDEN) return;
        
        // Get average activations
        auto avgActivations = nn.getAverageActivations();
        
        // Collect active neurons with their activation levels
        std::vector<std::pair<double, int>> neuronActivations;
        for (int i = 0; i < nn.hiddenSize; i++) {
            if (nn.neuronActive[i]) {
                // Also consider neurons that are always near 0 or always near 1 (saturated)
                // as less useful - they don't provide gradient signal
                double act = avgActivations[i];
                double sparsityScore = std::abs(act - 0.5);  // Distance from 0.5
                // Neurons near 0.5 are more useful (better gradient flow)
                // High sparsityScore = saturated (bad), Low = balanced (good)
                neuronActivations.push_back({sparsityScore, i});
            }
        }
        
        // Sort by sparsity score (highest first = most saturated = eliminate first)
        std::sort(neuronActivations.begin(), neuronActivations.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Eliminate most saturated neurons
        int toEliminate = std::max(1, static_cast<int>(neuronActivations.size() * eliminationRate));
        toEliminate = std::min(toEliminate, activeCount - MIN_HIDDEN);
        
        for (int i = 0; i < toEliminate; i++) {
            int neuronIdx = neuronActivations[i].second;
            nn.neuronActive[neuronIdx] = false;
            
            // Zero out the weights
            for (int j = 0; j < INPUT_SIZE; j++) {
                nn.weights_ih[neuronIdx][j] = 0.0;
            }
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                nn.weights_ho[j][neuronIdx] = 0.0;
            }
            nn.bias_h[neuronIdx] = 0.0;
        }
    }
    
    void train(DynamicNeuralNetwork& nn, const std::vector<DataPoint>& data, 
               int epoch, ProgressBar* progress) {
        // Forward pass with activation tracking
        for (const auto& dp : data) {
            nn.forward(dp.input, true);  // Track activations
        }
        
        // Regular gradient descent
        gdTrainer.train(nn, data, LEARNING_RATE);
        
        // Periodic elimination
        if ((epoch + 1) % ELIMINATION_INTERVAL == 0 && epoch < MAX_EPOCHS - 50) {
            eliminateLeastActiveNeurons(nn);
            nn.resetActivationTracking();
        }
        
        if (progress && epoch % 10 == 0) {
            progress->update(epoch);
        }
    }
};

// ============================================================================
// TRAINING METRICS
// ============================================================================

struct TrainingMetrics {
    double trainingTime;
    double finalAccuracy;
    double finalLoss;
    int activeNeurons;
};

// ============================================================================
// RUN TRAINING FUNCTIONS
// ============================================================================

TrainingMetrics trainGradientDescent(const std::vector<DataPoint>& trainData,
                                      const std::vector<DataPoint>& testData,
                                      ProgressBar* progress) {
    TrainingMetrics metrics;
    DynamicNeuralNetwork nn(8);  // Fixed 8 neurons for baseline
    GradientDescentTrainer trainer;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        trainer.train(nn, trainData, LEARNING_RATE);
        if (progress && epoch % 10 == 0) progress->update(epoch);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    metrics.trainingTime = std::chrono::duration<double, std::milli>(end - start).count();
    metrics.finalAccuracy = nn.calculateAccuracy(testData);
    metrics.finalLoss = nn.calculateLoss(testData);
    metrics.activeNeurons = 8;
    
    return metrics;
}

TrainingMetrics trainWithActivationElimination(const std::vector<DataPoint>& trainData,
                                                const std::vector<DataPoint>& testData,
                                                double elimRate, ProgressBar* progress) {
    TrainingMetrics metrics;
    DynamicNeuralNetwork nn(INITIAL_HIDDEN);  // Start with 24 neurons
    ActivationSparsityTrainer trainer(elimRate);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        trainer.train(nn, trainData, epoch, progress);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    metrics.trainingTime = std::chrono::duration<double, std::milli>(end - start).count();
    metrics.finalAccuracy = nn.calculateAccuracy(testData);
    metrics.finalLoss = nn.calculateLoss(testData);
    metrics.activeNeurons = nn.countActiveNeurons();
    
    return metrics;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       NEURAL NETWORK TRAINING #8: ACTIVATION SPARSITY ELIMINATION                         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Strategy: Eliminate neurons with saturated/sparse activations\n";
    std::cout << "  • Start with " << INITIAL_HIDDEN << " hidden neurons\n";
    std::cout << "  • Track average activation per neuron\n";
    std::cout << "  • Neurons near 0 or 1 (saturated) = poor gradient flow = eliminate\n";
    std::cout << "  • N=3: Eliminate 33% most saturated (Quaylyn's optimal)\n";
    std::cout << "  • N=9: Eliminate 11% most saturated (conservative)\n";
    std::cout << "  • Minimum " << MIN_HIDDEN << " neurons preserved\n\n";
    
    std::cout << "Running " << NUM_TRIALS << " trials...\n\n";
    
    std::vector<TrainingMetrics> gdResults, n3Results, n9Results;
    
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        std::cout << "  ╭─ Trial " << (trial + 1) << "/" << NUM_TRIALS 
                  << " ───────────────────────────────────────────────────────────────╮\n";
        
        auto trainData = generateDataset(TRAINING_SAMPLES, 0.1);
        auto testData = generateDataset(TEST_SAMPLES, 0.1);
        
        // Gradient Descent (fixed 8 neurons)
        ProgressBar gdProgress(MAX_EPOCHS, "Gradient (8 neu) ", 25);
        auto gdMetrics = trainGradientDescent(trainData, testData, &gdProgress);
        gdProgress.finish("→ " + std::to_string(int(gdMetrics.finalAccuracy * 100)) + 
                         "% (" + std::to_string(gdMetrics.activeNeurons) + " neu)");
        gdResults.push_back(gdMetrics);
        
        // N=3 (33% elimination)
        ProgressBar n3Progress(MAX_EPOCHS, "N=3 (33% elim)   ", 25);
        auto n3Metrics = trainWithActivationElimination(trainData, testData, 
                                                         ELIMINATION_RATE_N3, &n3Progress);
        n3Progress.finish("→ " + std::to_string(int(n3Metrics.finalAccuracy * 100)) + 
                         "% (" + std::to_string(n3Metrics.activeNeurons) + " neu)");
        n3Results.push_back(n3Metrics);
        
        // N=9 (11% elimination)
        ProgressBar n9Progress(MAX_EPOCHS, "N=9 (11% elim)   ", 25);
        auto n9Metrics = trainWithActivationElimination(trainData, testData, 
                                                         ELIMINATION_RATE_N9, &n9Progress);
        n9Progress.finish("→ " + std::to_string(int(n9Metrics.finalAccuracy * 100)) + 
                         "% (" + std::to_string(n9Metrics.activeNeurons) + " neu)");
        n9Results.push_back(n9Metrics);
        
        std::cout << "  ╰───────────────────────────────────────────────────────────────────────────────╯\n\n";
    }
    
    // Calculate averages
    auto average = [](const std::vector<TrainingMetrics>& v) {
        TrainingMetrics avg = {0, 0, 0, 0};
        for (const auto& m : v) {
            avg.trainingTime += m.trainingTime;
            avg.finalAccuracy += m.finalAccuracy;
            avg.finalLoss += m.finalLoss;
            avg.activeNeurons += m.activeNeurons;
        }
        int n = v.size();
        avg.trainingTime /= n;
        avg.finalAccuracy /= n;
        avg.finalLoss /= n;
        avg.activeNeurons /= n;
        return avg;
    };
    
    auto gdAvg = average(gdResults);
    auto n3Avg = average(n3Results);
    auto n9Avg = average(n9Results);
    
    // Print results
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                              RESULTS: ACTIVATION SPARSITY ELIMINATION                                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Metric                    │ Gradient (8 neu) │ N=3 (33% elim)   │ N=9 (11% elim)   │ Winner\n";
    std::cout << "  ──────────────────────────┼──────────────────┼──────────────────┼──────────────────┼────────────\n";
    
    // Training time
    std::cout << "  Training Time (ms)        │ " << std::setw(16) << gdAvg.trainingTime
              << " │ " << std::setw(16) << n3Avg.trainingTime
              << " │ " << std::setw(16) << n9Avg.trainingTime << " │ ";
    if (gdAvg.trainingTime <= n3Avg.trainingTime && gdAvg.trainingTime <= n9Avg.trainingTime) {
        std::cout << "Gradient\n";
    } else if (n3Avg.trainingTime <= n9Avg.trainingTime) {
        std::cout << "N=3\n";
    } else {
        std::cout << "N=9\n";
    }
    
    // Accuracy
    std::cout << "  Final Accuracy (%)        │ " << std::setw(15) << (gdAvg.finalAccuracy * 100) << "%"
              << " │ " << std::setw(15) << (n3Avg.finalAccuracy * 100) << "%"
              << " │ " << std::setw(15) << (n9Avg.finalAccuracy * 100) << "% │ ";
    if (gdAvg.finalAccuracy >= n3Avg.finalAccuracy && gdAvg.finalAccuracy >= n9Avg.finalAccuracy) {
        std::cout << "Gradient\n";
    } else if (n3Avg.finalAccuracy >= n9Avg.finalAccuracy) {
        std::cout << "N=3 (+" << std::setprecision(1) << ((n3Avg.finalAccuracy - gdAvg.finalAccuracy) * 100) << "%)\n";
    } else {
        std::cout << "N=9 (+" << std::setprecision(1) << ((n9Avg.finalAccuracy - gdAvg.finalAccuracy) * 100) << "%)\n";
    }
    std::cout << std::setprecision(2);
    
    // Loss
    std::cout << "  Final Loss                │ " << std::setw(16) << std::setprecision(4) << gdAvg.finalLoss
              << " │ " << std::setw(16) << n3Avg.finalLoss
              << " │ " << std::setw(16) << n9Avg.finalLoss << " │ ";
    if (gdAvg.finalLoss <= n3Avg.finalLoss && gdAvg.finalLoss <= n9Avg.finalLoss) {
        std::cout << "Gradient\n";
    } else if (n3Avg.finalLoss <= n9Avg.finalLoss) {
        std::cout << "N=3\n";
    } else {
        std::cout << "N=9\n";
    }
    
    // Active neurons
    std::cout << "  Final Neurons             │ " << std::setw(16) << gdAvg.activeNeurons
              << " │ " << std::setw(16) << n3Avg.activeNeurons
              << " │ " << std::setw(16) << n9Avg.activeNeurons << " │ Auto-discovered\n";
    
    std::cout << "\n  ACTIVATION SPARSITY PRINCIPLE:\n";
    std::cout << "  • Neurons with saturated activations (near 0 or 1) have vanishing gradients\n";
    std::cout << "  • They contribute little to learning → safe to eliminate\n";
    std::cout << "  • N=3 (33%): Aggressive elimination = compact model\n";
    std::cout << "  • N=9 (11%): Conservative = retains more capacity\n";
    
    double n3AccDiff = (n3Avg.finalAccuracy - gdAvg.finalAccuracy) * 100;
    double n9AccDiff = (n9Avg.finalAccuracy - gdAvg.finalAccuracy) * 100;
    
    std::cout << "\n  N=3 vs Gradient: " << std::showpos << std::setprecision(1) << n3AccDiff 
              << std::noshowpos << "% accuracy, " << n3Avg.activeNeurons << " final neurons\n";
    std::cout << "  N=9 vs Gradient: " << std::showpos << n9AccDiff 
              << std::noshowpos << "% accuracy, " << n9Avg.activeNeurons << " final neurons\n";
    
    std::cout << "\n";
    
    return 0;
}
