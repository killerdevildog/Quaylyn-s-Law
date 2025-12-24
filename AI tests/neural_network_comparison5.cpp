/**
 * Neural Network Training Comparison #5: LAYER-WISE ELIMINATION
 * 
 * Strategy: Eliminate worst neurons (not networks, not weights)
 * - Evaluate each neuron's contribution to output
 * - Eliminate bottom 33% of neurons by contribution
 * - Remaining neurons adapt to compensate
 * 
 * This is neural architecture search via elimination
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

constexpr int INITIAL_HIDDEN_NEURONS = 24;  // Start large, eliminate down
constexpr int MIN_HIDDEN_NEURONS = 4;       // Don't go below this
constexpr int INPUT_SIZE = 4;
constexpr int OUTPUT_SIZE = 2;
constexpr int TRAINING_SAMPLES = 1000;
constexpr int TEST_SAMPLES = 200;
constexpr int MAX_EPOCHS = 500;
constexpr double LEARNING_RATE = 0.1;
constexpr double TARGET_ACCURACY = 0.95;
constexpr int NUM_TRIALS = 10;

// Elimination settings
constexpr double ELIMINATION_RATE_N3 = 0.33;  // Eliminate 33% of neurons (N=3)
constexpr double ELIMINATION_RATE_N9 = 0.11;  // Eliminate 11% of neurons (N=9)
constexpr int ELIMINATION_INTERVAL = 100;     // Every N epochs

// For comparison baseline
constexpr int BASELINE_HIDDEN = 8;

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
// DYNAMIC NEURAL NETWORK (variable hidden size)
// ============================================================================

struct DynamicNeuralNetwork {
    int hiddenSize;
    std::vector<std::vector<double>> weights_ih, weights_ho;
    std::vector<double> bias_h, bias_o;
    std::vector<double> hidden_raw, hidden_act, output_raw, output_act;
    std::vector<bool> neuronActive;  // Which neurons are still active
    
    DynamicNeuralNetwork(int hidden = INITIAL_HIDDEN_NEURONS) : hiddenSize(hidden) {
        weights_ih.resize(hiddenSize, std::vector<double>(INPUT_SIZE));
        weights_ho.resize(OUTPUT_SIZE, std::vector<double>(hiddenSize));
        bias_h.resize(hiddenSize);
        bias_o.resize(OUTPUT_SIZE);
        hidden_raw.resize(hiddenSize);
        hidden_act.resize(hiddenSize);
        output_raw.resize(OUTPUT_SIZE);
        output_act.resize(OUTPUT_SIZE);
        neuronActive.resize(hiddenSize, true);
        randomizeWeights();
    }
    
    void randomizeWeights() {
        double range = std::sqrt(6.0 / (INPUT_SIZE + hiddenSize));
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) weights_ih[i][j] = randomDouble(-range, range);
            bias_h[i] = randomDouble(-0.1, 0.1);
        }
        range = std::sqrt(6.0 / (hiddenSize + OUTPUT_SIZE));
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < hiddenSize; j++) weights_ho[i][j] = randomDouble(-range, range);
            bias_o[i] = randomDouble(-0.1, 0.1);
        }
    }
    
    int getActiveNeuronCount() {
        int count = 0;
        for (bool a : neuronActive) if (a) count++;
        return count;
    }
    
    std::vector<double> forward(const std::vector<double>& input) {
        for (int i = 0; i < hiddenSize; i++) {
            if (!neuronActive[i]) {
                hidden_act[i] = 0.0;
                continue;
            }
            hidden_raw[i] = bias_h[i];
            for (int j = 0; j < INPUT_SIZE; j++) hidden_raw[i] += weights_ih[i][j] * input[j];
            hidden_act[i] = sigmoid(hidden_raw[i]);
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output_raw[i] = bias_o[i];
            for (int j = 0; j < hiddenSize; j++) {
                if (neuronActive[j]) output_raw[i] += weights_ho[i][j] * hidden_act[j];
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
    
    // Evaluate each neuron's contribution
    std::vector<double> evaluateNeuronImportance(const std::vector<DataPoint>& valData) {
        std::vector<double> importance(hiddenSize, 0.0);
        double baseLoss = calculateLoss(valData);
        
        for (int n = 0; n < hiddenSize; n++) {
            if (!neuronActive[n]) {
                importance[n] = -1e9;  // Already eliminated
                continue;
            }
            
            // Temporarily disable neuron
            neuronActive[n] = false;
            double lossWithout = calculateLoss(valData);
            neuronActive[n] = true;
            
            // Importance = how much loss increases without this neuron
            importance[n] = lossWithout - baseLoss;
        }
        
        return importance;
    }
    
    // Eliminate least important neurons
    void eliminateWeakNeurons(const std::vector<DataPoint>& valData, double rate) {
        int activeCount = getActiveNeuronCount();
        if (activeCount <= MIN_HIDDEN_NEURONS) return;
        
        auto importance = evaluateNeuronImportance(valData);
        
        // Sort neurons by importance (least important first)
        std::vector<std::pair<double, int>> ranked;
        for (int i = 0; i < hiddenSize; i++) {
            if (neuronActive[i]) {
                ranked.push_back({importance[i], i});
            }
        }
        std::sort(ranked.begin(), ranked.end());
        
        // Eliminate bottom 33%
        int numToEliminate = static_cast<int>(activeCount * rate);
        numToEliminate = std::min(numToEliminate, activeCount - MIN_HIDDEN_NEURONS);
        
        for (int k = 0; k < numToEliminate; k++) {
            int idx = ranked[k].second;
            neuronActive[idx] = false;
        }
    }
};

// ============================================================================
// TRAINERS
// ============================================================================

struct GradientDescentTrainer {
    void train(DynamicNeuralNetwork& nn, const std::vector<DataPoint>& data, double lr) {
        std::vector<std::vector<double>> grad_ih(nn.hiddenSize, std::vector<double>(INPUT_SIZE, 0.0));
        std::vector<std::vector<double>> grad_ho(OUTPUT_SIZE, std::vector<double>(nn.hiddenSize, 0.0));
        std::vector<double> grad_bh(nn.hiddenSize, 0.0), grad_bo(OUTPUT_SIZE, 0.0);
        
        for (const auto& dp : data) {
            nn.forward(dp.input);
            std::vector<double> output_delta(OUTPUT_SIZE);
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                output_delta[i] = (nn.output_act[i] - dp.target[i]) * sigmoidDerivative(nn.output_raw[i]);
            }
            std::vector<double> hidden_delta(nn.hiddenSize);
            for (int i = 0; i < nn.hiddenSize; i++) {
                if (!nn.neuronActive[i]) continue;
                double error = 0.0;
                for (int j = 0; j < OUTPUT_SIZE; j++) error += output_delta[j] * nn.weights_ho[j][i];
                hidden_delta[i] = error * sigmoidDerivative(nn.hidden_raw[i]);
            }
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                for (int j = 0; j < nn.hiddenSize; j++) {
                    if (nn.neuronActive[j]) grad_ho[i][j] += output_delta[i] * nn.hidden_act[j];
                }
                grad_bo[i] += output_delta[i];
            }
            for (int i = 0; i < nn.hiddenSize; i++) {
                if (!nn.neuronActive[i]) continue;
                for (int j = 0; j < INPUT_SIZE; j++) grad_ih[i][j] += hidden_delta[i] * dp.input[j];
                grad_bh[i] += hidden_delta[i];
            }
        }
        
        double scale = lr / data.size();
        for (int i = 0; i < nn.hiddenSize; i++) {
            if (!nn.neuronActive[i]) continue;
            for (int j = 0; j < INPUT_SIZE; j++) nn.weights_ih[i][j] -= scale * grad_ih[i][j];
            nn.bias_h[i] -= scale * grad_bh[i];
        }
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < nn.hiddenSize; j++) {
                if (nn.neuronActive[j]) nn.weights_ho[i][j] -= scale * grad_ho[i][j];
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
    int finalNeurons;
    bool converged;
};

TrainingMetrics trainBaseline(const std::vector<DataPoint>& trainData,
                               const std::vector<DataPoint>& testData,
                               ProgressBar* progress = nullptr) {
    TrainingMetrics metrics;
    DynamicNeuralNetwork nn(BASELINE_HIDDEN);  // Fixed 8 neurons
    GradientDescentTrainer trainer;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        trainer.train(nn, trainData, LEARNING_RATE);
        if (progress && epoch % 10 == 0) progress->update(epoch);
    }
    if (progress) progress->update(MAX_EPOCHS);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    metrics.trainingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    metrics.finalAccuracy = nn.calculateAccuracy(testData);
    metrics.finalLoss = nn.calculateLoss(testData);
    metrics.finalNeurons = nn.getActiveNeuronCount();
    metrics.converged = metrics.finalAccuracy >= TARGET_ACCURACY;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) nn.predict(testData[i % testData.size()].input);
    auto t2 = std::chrono::high_resolution_clock::now();
    metrics.inferenceTime = std::chrono::duration<double, std::micro>(t2 - t1).count() / 1000.0;
    
    return metrics;
}

TrainingMetrics trainWithNeuronElimination(const std::vector<DataPoint>& trainData,
                                            const std::vector<DataPoint>& testData,
                                            double eliminationRate,
                                            ProgressBar* progress = nullptr) {
    TrainingMetrics metrics;
    DynamicNeuralNetwork nn(INITIAL_HIDDEN_NEURONS);  // Start with 24
    GradientDescentTrainer trainer;
    
    std::vector<DataPoint> valData(trainData.begin(), trainData.begin() + trainData.size() / 5);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        trainer.train(nn, trainData, LEARNING_RATE);
        
        // Eliminate neurons periodically
        if (epoch > 0 && epoch % ELIMINATION_INTERVAL == 0) {
            nn.eliminateWeakNeurons(valData, eliminationRate);
        }
        
        if (progress && epoch % 10 == 0) progress->update(epoch);
    }
    if (progress) progress->update(MAX_EPOCHS);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    metrics.trainingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    metrics.finalAccuracy = nn.calculateAccuracy(testData);
    metrics.finalLoss = nn.calculateLoss(testData);
    metrics.finalNeurons = nn.getActiveNeuronCount();
    metrics.converged = metrics.finalAccuracy >= TARGET_ACCURACY;
    
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
    std::cout << "║       NEURAL NETWORK TRAINING #5: LAYER-WISE NEURON ELIMINATION                            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Strategy: Start large, eliminate least important neurons\n";
    std::cout << "  • Start with " << INITIAL_HIDDEN_NEURONS << " hidden neurons\n";
    std::cout << "  • Evaluate each neuron's contribution (ablation)\n";
    std::cout << "  • N=3: Eliminate 33% every " << ELIMINATION_INTERVAL << " epochs\n";
    std::cout << "  • N=9: Eliminate 11% every " << ELIMINATION_INTERVAL << " epochs\n";
    std::cout << "  • Minimum " << MIN_HIDDEN_NEURONS << " neurons preserved\n\n";
    
    std::vector<TrainingMetrics> gdMetrics, n3Metrics, n9Metrics;
    
    std::cout << "Running " << NUM_TRIALS << " trials...\n\n";
    
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        std::cout << "  ╭─ Trial " << (trial + 1) << "/" << NUM_TRIALS << " ───────────────────────────────────────────────────────────────╮\n";
        
        auto trainData = generateDataset(TRAINING_SAMPLES, 0.1);
        auto testData = generateDataset(TEST_SAMPLES, 0.1);
        
        ProgressBar gdProg(MAX_EPOCHS, "Gradient (8 neu) ", 25);
        auto gdResult = trainBaseline(trainData, testData, &gdProg);
        gdProg.finish("→ " + std::to_string(static_cast<int>(gdResult.finalAccuracy * 100)) + "% (" + 
                      std::to_string(gdResult.finalNeurons) + " neu)");
        gdMetrics.push_back(gdResult);
        
        ProgressBar n3Prog(MAX_EPOCHS, "N=3 (33% elim)   ", 25);
        auto n3Result = trainWithNeuronElimination(trainData, testData, ELIMINATION_RATE_N3, &n3Prog);
        n3Prog.finish("→ " + std::to_string(static_cast<int>(n3Result.finalAccuracy * 100)) + "% (" + 
                      std::to_string(n3Result.finalNeurons) + " neu)");
        n3Metrics.push_back(n3Result);
        
        ProgressBar n9Prog(MAX_EPOCHS, "N=9 (11% elim)   ", 25);
        auto n9Result = trainWithNeuronElimination(trainData, testData, ELIMINATION_RATE_N9, &n9Prog);
        n9Prog.finish("→ " + std::to_string(static_cast<int>(n9Result.finalAccuracy * 100)) + "% (" + 
                      std::to_string(n9Result.finalNeurons) + " neu)");
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
            a.finalNeurons += x.finalNeurons;
        }
        int n = m.size();
        a.trainingTime /= n; a.finalAccuracy /= n; a.finalLoss /= n; 
        a.inferenceTime /= n; a.finalNeurons /= n;
        return a;
    };
    
    auto gdAvg = avg(gdMetrics), n3Avg = avg(n3Metrics), n9Avg = avg(n9Metrics);
    
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                              RESULTS: NEURON ELIMINATION                                               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Metric                    │ Gradient (8 neu) │ N=3 (33% elim)   │ N=9 (11% elim)   │ Winner\n";
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
    std::cout << "  Final Neurons             │ " << std::setw(16) << gdAvg.finalNeurons 
              << " │ " << std::setw(16) << n3Avg.finalNeurons 
              << " │ " << std::setw(16) << n9Avg.finalNeurons << " │ Auto-discovered\n";
    
    std::cout << "\n  NEURON ELIMINATION PRINCIPLE:\n";
    std::cout << "  • N=3: Aggressive (33%) = finds minimal architecture\n";
    std::cout << "  • N=9: Conservative (11%) = retains more capacity\n";
    std::cout << "  • Start over-parameterized, let elimination find optimal size\n";
    
    double n3VsGd = (n3Avg.finalAccuracy - gdAvg.finalAccuracy) * 100;
    double n9VsGd = (n9Avg.finalAccuracy - gdAvg.finalAccuracy) * 100;
    
    std::cout << std::setprecision(1);
    std::cout << "\n  N=3 vs Gradient: " << (n3VsGd >= 0 ? "+" : "") << n3VsGd << "% accuracy, ";
    std::cout << n3Avg.finalNeurons << " final neurons\n";
    std::cout << "  N=9 vs Gradient: " << (n9VsGd >= 0 ? "+" : "") << n9VsGd << "% accuracy, ";
    std::cout << n9Avg.finalNeurons << " final neurons\n";
    
    std::cout << "\n";
    return 0;
}
