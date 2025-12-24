/**
 * Neural Network Training Comparison #7: GRADIENT MAGNITUDE ELIMINATION
 * 
 * Strategy: Eliminate weights with smallest gradient magnitudes
 *   - Gradients indicate how much a weight affects the loss
 *   - Small gradients = weight doesn't matter much = can be eliminated
 *   - N=3: Eliminate 33% smallest gradients (Quaylyn's optimal)
 *   - N=9: Eliminate 11% smallest gradients (conservative)
 * 
 * This tests Quaylyn's Law on IMPORTANCE-BASED elimination
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
constexpr int NUM_TRIALS = 10;

// Quaylyn's Law configurations
constexpr double ELIMINATION_RATE_N3 = 0.33;  // 33% elimination (optimal)
constexpr double ELIMINATION_RATE_N9 = 0.11;  // 11% elimination (conservative)
constexpr int ELIMINATION_INTERVAL = 50;       // Eliminate every N epochs

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
// NEURAL NETWORK WITH GRADIENT TRACKING
// ============================================================================

struct NeuralNetwork {
    std::vector<std::vector<double>> weights_ih;
    std::vector<std::vector<double>> weights_ho;
    std::vector<double> bias_h, bias_o;
    std::vector<double> hidden_raw, hidden_act, output_raw, output_act;
    
    // Gradient accumulators for magnitude tracking
    std::vector<std::vector<double>> grad_ih_accum;
    std::vector<std::vector<double>> grad_ho_accum;
    std::vector<double> grad_bh_accum, grad_bo_accum;
    
    // Mask for eliminated weights (0 = eliminated, 1 = active)
    std::vector<std::vector<double>> mask_ih;
    std::vector<std::vector<double>> mask_ho;
    
    NeuralNetwork() {
        weights_ih.resize(HIDDEN_NEURONS, std::vector<double>(INPUT_SIZE));
        weights_ho.resize(OUTPUT_SIZE, std::vector<double>(HIDDEN_NEURONS));
        bias_h.resize(HIDDEN_NEURONS);
        bias_o.resize(OUTPUT_SIZE);
        hidden_raw.resize(HIDDEN_NEURONS);
        hidden_act.resize(HIDDEN_NEURONS);
        output_raw.resize(OUTPUT_SIZE);
        output_act.resize(OUTPUT_SIZE);
        
        // Gradient accumulators
        grad_ih_accum.resize(HIDDEN_NEURONS, std::vector<double>(INPUT_SIZE, 0.0));
        grad_ho_accum.resize(OUTPUT_SIZE, std::vector<double>(HIDDEN_NEURONS, 0.0));
        grad_bh_accum.resize(HIDDEN_NEURONS, 0.0);
        grad_bo_accum.resize(OUTPUT_SIZE, 0.0);
        
        // Masks (all active initially)
        mask_ih.resize(HIDDEN_NEURONS, std::vector<double>(INPUT_SIZE, 1.0));
        mask_ho.resize(OUTPUT_SIZE, std::vector<double>(HIDDEN_NEURONS, 1.0));
        
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
    
    void resetGradientAccumulators() {
        for (auto& row : grad_ih_accum) std::fill(row.begin(), row.end(), 0.0);
        for (auto& row : grad_ho_accum) std::fill(row.begin(), row.end(), 0.0);
        std::fill(grad_bh_accum.begin(), grad_bh_accum.end(), 0.0);
        std::fill(grad_bo_accum.begin(), grad_bo_accum.end(), 0.0);
    }
    
    std::vector<double> forward(const std::vector<double>& input) {
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            hidden_raw[i] = bias_h[i];
            for (int j = 0; j < INPUT_SIZE; j++) {
                hidden_raw[i] += weights_ih[i][j] * mask_ih[i][j] * input[j];
            }
            hidden_act[i] = sigmoid(hidden_raw[i]);
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output_raw[i] = bias_o[i];
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                output_raw[i] += weights_ho[i][j] * mask_ho[i][j] * hidden_act[j];
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
    
    int countActiveWeights() {
        int count = 0;
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                if (mask_ih[i][j] > 0.5) count++;
            }
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                if (mask_ho[i][j] > 0.5) count++;
            }
        }
        return count;
    }
};

// ============================================================================
// GRADIENT DESCENT TRAINER
// ============================================================================

struct GradientDescentTrainer {
    void train(NeuralNetwork& nn, const std::vector<DataPoint>& data, double lr) {
        std::vector<std::vector<double>> grad_ih(HIDDEN_NEURONS, 
            std::vector<double>(INPUT_SIZE, 0.0));
        std::vector<std::vector<double>> grad_ho(OUTPUT_SIZE, 
            std::vector<double>(HIDDEN_NEURONS, 0.0));
        std::vector<double> grad_bh(HIDDEN_NEURONS, 0.0);
        std::vector<double> grad_bo(OUTPUT_SIZE, 0.0);
        
        for (const auto& dp : data) {
            nn.forward(dp.input);
            
            std::vector<double> output_delta(OUTPUT_SIZE);
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                double error = nn.output_act[i] - dp.target[i];
                output_delta[i] = error * sigmoidDeriv(nn.output_raw[i]);
            }
            
            std::vector<double> hidden_delta(HIDDEN_NEURONS);
            for (int i = 0; i < HIDDEN_NEURONS; i++) {
                double error = 0.0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    error += output_delta[j] * nn.weights_ho[j][i];
                }
                hidden_delta[i] = error * sigmoidDeriv(nn.hidden_raw[i]);
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
        
        double scale = lr / data.size();
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
// GRADIENT MAGNITUDE ELIMINATION TRAINER
// ============================================================================

struct GradientMagnitudeTrainer {
    double eliminationRate;
    GradientDescentTrainer gdTrainer;
    
    GradientMagnitudeTrainer(double elimRate) : eliminationRate(elimRate) {}
    
    void computeAndAccumulateGradients(NeuralNetwork& nn, const std::vector<DataPoint>& data) {
        for (const auto& dp : data) {
            nn.forward(dp.input);
            
            std::vector<double> output_delta(OUTPUT_SIZE);
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                double error = nn.output_act[i] - dp.target[i];
                output_delta[i] = error * sigmoidDeriv(nn.output_raw[i]);
            }
            
            std::vector<double> hidden_delta(HIDDEN_NEURONS);
            for (int i = 0; i < HIDDEN_NEURONS; i++) {
                double error = 0.0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    error += output_delta[j] * nn.weights_ho[j][i];
                }
                hidden_delta[i] = error * sigmoidDeriv(nn.hidden_raw[i]);
            }
            
            // Accumulate absolute gradients
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                for (int j = 0; j < HIDDEN_NEURONS; j++) {
                    nn.grad_ho_accum[i][j] += std::abs(output_delta[i] * nn.hidden_act[j]);
                }
                nn.grad_bo_accum[i] += std::abs(output_delta[i]);
            }
            
            for (int i = 0; i < HIDDEN_NEURONS; i++) {
                for (int j = 0; j < INPUT_SIZE; j++) {
                    nn.grad_ih_accum[i][j] += std::abs(hidden_delta[i] * dp.input[j]);
                }
                nn.grad_bh_accum[i] += std::abs(hidden_delta[i]);
            }
        }
    }
    
    void eliminateSmallestGradients(NeuralNetwork& nn) {
        // Collect all gradient magnitudes with their indices
        std::vector<std::tuple<double, int, int, int>> gradients; // (magnitude, layer, i, j)
        
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                if (nn.mask_ih[i][j] > 0.5) {  // Only consider active weights
                    gradients.push_back({nn.grad_ih_accum[i][j], 0, i, j});
                }
            }
        }
        
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                if (nn.mask_ho[i][j] > 0.5) {
                    gradients.push_back({nn.grad_ho_accum[i][j], 1, i, j});
                }
            }
        }
        
        if (gradients.empty()) return;
        
        // Sort by gradient magnitude (ascending - smallest first)
        std::sort(gradients.begin(), gradients.end(),
            [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
        
        // Eliminate smallest gradients
        int toEliminate = std::max(1, static_cast<int>(gradients.size() * eliminationRate));
        toEliminate = std::min(toEliminate, static_cast<int>(gradients.size()) - 4); // Keep at least 4
        
        for (int i = 0; i < toEliminate; i++) {
            auto [mag, layer, row, col] = gradients[i];
            if (layer == 0) {
                nn.mask_ih[row][col] = 0.0;
                nn.weights_ih[row][col] = 0.0;
            } else {
                nn.mask_ho[row][col] = 0.0;
                nn.weights_ho[row][col] = 0.0;
            }
        }
    }
    
    void train(NeuralNetwork& nn, const std::vector<DataPoint>& data, 
               int epoch, ProgressBar* progress) {
        // Regular gradient descent step
        gdTrainer.train(nn, data, LEARNING_RATE);
        
        // Accumulate gradients
        computeAndAccumulateGradients(nn, data);
        
        // Periodic elimination
        if ((epoch + 1) % ELIMINATION_INTERVAL == 0 && epoch < MAX_EPOCHS - 50) {
            eliminateSmallestGradients(nn);
            nn.resetGradientAccumulators();
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
    int activeWeights;
};

// ============================================================================
// RUN TRAINING FUNCTIONS
// ============================================================================

TrainingMetrics trainGradientDescent(const std::vector<DataPoint>& trainData,
                                      const std::vector<DataPoint>& testData,
                                      ProgressBar* progress) {
    TrainingMetrics metrics;
    NeuralNetwork nn;
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
    metrics.activeWeights = HIDDEN_NEURONS * INPUT_SIZE + OUTPUT_SIZE * HIDDEN_NEURONS;
    
    return metrics;
}

TrainingMetrics trainWithGradientElimination(const std::vector<DataPoint>& trainData,
                                              const std::vector<DataPoint>& testData,
                                              double elimRate, ProgressBar* progress) {
    TrainingMetrics metrics;
    NeuralNetwork nn;
    GradientMagnitudeTrainer trainer(elimRate);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        trainer.train(nn, trainData, epoch, progress);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    metrics.trainingTime = std::chrono::duration<double, std::milli>(end - start).count();
    metrics.finalAccuracy = nn.calculateAccuracy(testData);
    metrics.finalLoss = nn.calculateLoss(testData);
    metrics.activeWeights = nn.countActiveWeights();
    
    return metrics;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       NEURAL NETWORK TRAINING #7: GRADIENT MAGNITUDE ELIMINATION                           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Strategy: Eliminate weights with smallest gradient magnitudes\n";
    std::cout << "  • Small gradient = weight has little effect on loss\n";
    std::cout << "  • N=3: Eliminate 33% smallest gradients (Quaylyn's optimal)\n";
    std::cout << "  • N=9: Eliminate 11% smallest gradients (conservative)\n";
    std::cout << "  • Elimination every " << ELIMINATION_INTERVAL << " epochs\n\n";
    
    std::cout << "Running " << NUM_TRIALS << " trials...\n\n";
    
    std::vector<TrainingMetrics> gdResults, n3Results, n9Results;
    
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        std::cout << "  ╭─ Trial " << (trial + 1) << "/" << NUM_TRIALS 
                  << " ───────────────────────────────────────────────────────────────╮\n";
        
        auto trainData = generateDataset(TRAINING_SAMPLES, 0.1);
        auto testData = generateDataset(TEST_SAMPLES, 0.1);
        
        // Gradient Descent
        ProgressBar gdProgress(MAX_EPOCHS, "Gradient Descent ", 25);
        auto gdMetrics = trainGradientDescent(trainData, testData, &gdProgress);
        gdProgress.finish("→ " + std::to_string(int(gdMetrics.finalAccuracy * 100)) + 
                         "% (" + std::to_string(gdMetrics.activeWeights) + " wts)");
        gdResults.push_back(gdMetrics);
        
        // N=3 (33% elimination)
        ProgressBar n3Progress(MAX_EPOCHS, "N=3 (33% elim)   ", 25);
        auto n3Metrics = trainWithGradientElimination(trainData, testData, 
                                                       ELIMINATION_RATE_N3, &n3Progress);
        n3Progress.finish("→ " + std::to_string(int(n3Metrics.finalAccuracy * 100)) + 
                         "% (" + std::to_string(n3Metrics.activeWeights) + " wts)");
        n3Results.push_back(n3Metrics);
        
        // N=9 (11% elimination)
        ProgressBar n9Progress(MAX_EPOCHS, "N=9 (11% elim)   ", 25);
        auto n9Metrics = trainWithGradientElimination(trainData, testData, 
                                                       ELIMINATION_RATE_N9, &n9Progress);
        n9Progress.finish("→ " + std::to_string(int(n9Metrics.finalAccuracy * 100)) + 
                         "% (" + std::to_string(n9Metrics.activeWeights) + " wts)");
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
            avg.activeWeights += m.activeWeights;
        }
        int n = v.size();
        avg.trainingTime /= n;
        avg.finalAccuracy /= n;
        avg.finalLoss /= n;
        avg.activeWeights /= n;
        return avg;
    };
    
    auto gdAvg = average(gdResults);
    auto n3Avg = average(n3Results);
    auto n9Avg = average(n9Results);
    
    // Print results
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                              RESULTS: GRADIENT MAGNITUDE ELIMINATION                                   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Metric                    │ Gradient Descent │ N=3 (33% elim)   │ N=9 (11% elim)   │ Winner\n";
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
    
    // Active weights
    std::cout << "  Active Weights            │ " << std::setw(16) << gdAvg.activeWeights
              << " │ " << std::setw(16) << n3Avg.activeWeights
              << " │ " << std::setw(16) << n9Avg.activeWeights << " │ Smaller=Better\n";
    
    std::cout << "\n  GRADIENT MAGNITUDE ELIMINATION PRINCIPLE:\n";
    std::cout << "  • Weights with small gradients contribute little to learning\n";
    std::cout << "  • Eliminating them reduces model complexity without losing accuracy\n";
    std::cout << "  • N=3 (33%): Aggressive pruning based on importance\n";
    std::cout << "  • N=9 (11%): Conservative pruning retains more capacity\n";
    
    double n3AccDiff = (n3Avg.finalAccuracy - gdAvg.finalAccuracy) * 100;
    double n9AccDiff = (n9Avg.finalAccuracy - gdAvg.finalAccuracy) * 100;
    
    std::cout << "\n  N=3 vs Gradient: " << std::showpos << std::setprecision(1) << n3AccDiff 
              << std::noshowpos << "% accuracy, " << n3Avg.activeWeights << "/" << gdAvg.activeWeights << " weights\n";
    std::cout << "  N=9 vs Gradient: " << std::showpos << n9AccDiff 
              << std::noshowpos << "% accuracy, " << n9Avg.activeWeights << "/" << gdAvg.activeWeights << " weights\n";
    
    std::cout << "\n";
    
    return 0;
}
