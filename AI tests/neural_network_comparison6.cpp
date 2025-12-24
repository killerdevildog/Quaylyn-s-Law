/**
 * Neural Network Training Comparison #6: ENSEMBLE DISAGREEMENT ELIMINATION
 * 
 * Strategy: Maintain ensemble, eliminate members that disagree with majority
 * - Train N=9 networks in parallel
 * - On each sample, networks vote on classification
 * - Networks that consistently disagree with consensus are eliminated
 * - Replaced with networks that agree (via mutation)
 * 
 * Advantage: Eliminates overconfident wrong networks
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <map>

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

// Ensemble settings
constexpr int ENSEMBLE_SIZE = 9;            // N=9 networks
constexpr int ELIMINATION_COUNT_N3 = 3;     // Eliminate 3/9 = 33% (N=3)
constexpr int ELIMINATION_COUNT_N9 = 1;     // Eliminate 1/9 = 11% (N=9)
constexpr double MUTATION_RATE = 0.3;
constexpr double MUTATION_STRENGTH = 0.5;
constexpr int VOTING_BATCH_SIZE = 50;       // Samples to vote on
constexpr int ELIMINATION_INTERVAL = 25;    // Epochs between eliminations

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
    
    double getConfidence(const std::vector<double>& input) {
        auto output = forward(input);
        return std::abs(output[0] - output[1]);
    }
    
    double calculateAccuracy(const std::vector<DataPoint>& data) {
        int correct = 0;
        for (const auto& dp : data) if (predict(dp.input) == dp.label) correct++;
        return static_cast<double>(correct) / data.size();
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
// ENSEMBLE DISAGREEMENT ELIMINATION TRAINER
// ============================================================================

struct EnsembleDisagreementTrainer {
    std::vector<NeuralNetwork> ensemble;
    GradientDescentTrainer gdTrainer;
    int eliminationCount;
    
    EnsembleDisagreementTrainer(int elimCount) : ensemble(ENSEMBLE_SIZE), eliminationCount(elimCount) {}
    
    // Get ensemble's majority vote
    int getConsensus(const std::vector<double>& input) {
        std::map<int, int> votes;
        for (auto& nn : ensemble) {
            int pred = nn.predict(input);
            votes[pred]++;
        }
        int bestVote = 0, bestCount = 0;
        for (auto& [vote, count] : votes) {
            if (count > bestCount) {
                bestCount = count;
                bestVote = vote;
            }
        }
        return bestVote;
    }
    
    // Evaluate disagreement with consensus
    std::vector<double> evaluateDisagreement(const std::vector<DataPoint>& batch) {
        std::vector<double> disagreement(ENSEMBLE_SIZE, 0.0);
        
        for (const auto& dp : batch) {
            int consensus = getConsensus(dp.input);
            
            for (int i = 0; i < ENSEMBLE_SIZE; i++) {
                int pred = ensemble[i].predict(dp.input);
                if (pred != consensus) {
                    // Penalize more if confident and wrong
                    double conf = ensemble[i].getConfidence(dp.input);
                    disagreement[i] += 1.0 + conf;
                }
            }
        }
        
        return disagreement;
    }
    
    NeuralNetwork train(const std::vector<DataPoint>& trainData,
                        const std::vector<DataPoint>& valData,
                        int maxEpochs,
                        ProgressBar* prog = nullptr) {
        
        // Train all ensemble members with gradient descent
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            // Train each network
            for (auto& nn : ensemble) {
                gdTrainer.train(nn, trainData, LEARNING_RATE);
            }
            
            // Periodically eliminate disagreeing members
            if (epoch > 0 && epoch % ELIMINATION_INTERVAL == 0) {
                // Get batch for voting
                std::vector<DataPoint> batch(valData.begin(), 
                    valData.begin() + std::min(VOTING_BATCH_SIZE, (int)valData.size()));
                
                auto disagreement = evaluateDisagreement(batch);
                
                // Sort by disagreement (worst first)
                std::vector<std::pair<double, int>> ranked;
                for (int i = 0; i < ENSEMBLE_SIZE; i++) {
                    ranked.push_back({disagreement[i], i});
                }
                std::sort(ranked.begin(), ranked.end(), 
                    [](const auto& a, const auto& b) { return a.first > b.first; });
                
                // Eliminate worst (highest disagreement)
                std::vector<int> survivors;
                for (int i = eliminationCount; i < ENSEMBLE_SIZE; i++) {
                    survivors.push_back(ranked[i].second);
                }
                
                // Replace eliminated with mutated survivors
                std::uniform_int_distribution<int> survDist(0, survivors.size() - 1);
                for (int i = 0; i < eliminationCount; i++) {
                    int loserIdx = ranked[i].second;
                    int parentIdx = survivors[survDist(rng)];
                    ensemble[loserIdx].copyFrom(ensemble[parentIdx]);
                    ensemble[loserIdx].mutate(MUTATION_RATE, MUTATION_STRENGTH);
                }
            }
            
            if (prog && epoch % 5 == 0) prog->update(epoch);
        }
        
        if (prog) prog->update(maxEpochs);
        
        // Return best performing network
        NeuralNetwork best;
        double bestLoss = 1e9;
        for (auto& nn : ensemble) {
            double loss = nn.calculateLoss(valData);
            if (loss < bestLoss) {
                bestLoss = loss;
                best.copyFrom(nn);
            }
        }
        return best;
    }
    
    // Ensemble prediction (voting)
    int predictEnsemble(const std::vector<double>& input) {
        return getConsensus(input);
    }
    
    double calculateEnsembleAccuracy(const std::vector<DataPoint>& data) {
        int correct = 0;
        for (const auto& dp : data) {
            if (getConsensus(dp.input) == dp.label) correct++;
        }
        return static_cast<double>(correct) / data.size();
    }
};

// ============================================================================
// TRAINING METRICS & RUNNERS
// ============================================================================

struct TrainingMetrics {
    double trainingTime, finalAccuracy, ensembleAccuracy, inferenceTime, finalLoss;
    bool converged;
};

TrainingMetrics trainWithGradientDescent(const std::vector<DataPoint>& trainData,
                                          const std::vector<DataPoint>& testData,
                                          ProgressBar* progress = nullptr) {
    TrainingMetrics metrics;
    NeuralNetwork nn;
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
    metrics.ensembleAccuracy = metrics.finalAccuracy;  // Same for single network
    metrics.finalLoss = nn.calculateLoss(testData);
    metrics.converged = metrics.finalAccuracy >= TARGET_ACCURACY;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) nn.predict(testData[i % testData.size()].input);
    auto t2 = std::chrono::high_resolution_clock::now();
    metrics.inferenceTime = std::chrono::duration<double, std::micro>(t2 - t1).count() / 1000.0;
    
    return metrics;
}

TrainingMetrics trainWithEnsembleDisagreement(const std::vector<DataPoint>& trainData,
                                               const std::vector<DataPoint>& testData,
                                               int eliminationCount,
                                               ProgressBar* progress = nullptr) {
    TrainingMetrics metrics;
    EnsembleDisagreementTrainer trainer(eliminationCount);
    
    std::vector<DataPoint> valData(trainData.begin(), trainData.begin() + trainData.size() / 5);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    NeuralNetwork bestNet = trainer.train(trainData, valData, MAX_EPOCHS, progress);
    auto endTime = std::chrono::high_resolution_clock::now();
    
    metrics.trainingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    metrics.finalAccuracy = bestNet.calculateAccuracy(testData);
    metrics.ensembleAccuracy = trainer.calculateEnsembleAccuracy(testData);
    metrics.finalLoss = bestNet.calculateLoss(testData);
    metrics.converged = metrics.finalAccuracy >= TARGET_ACCURACY;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) bestNet.predict(testData[i % testData.size()].input);
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
    std::cout << "║       NEURAL NETWORK TRAINING #6: ENSEMBLE DISAGREEMENT ELIMINATION                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Strategy: Eliminate networks that disagree with ensemble consensus\n";
    std::cout << "  • Maintain " << ENSEMBLE_SIZE << " networks trained in parallel\n";
    std::cout << "  • Networks vote on each sample\n";
    std::cout << "  • N=3: Eliminate 3/9 (33%) that disagree most\n";
    std::cout << "  • N=9: Eliminate 1/9 (11%) that disagrees most\n";
    std::cout << "  • Penalize confident but wrong predictions\n\n";
    
    std::vector<TrainingMetrics> gdMetrics, n3Metrics, n9Metrics;
    
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
        auto n3Result = trainWithEnsembleDisagreement(trainData, testData, ELIMINATION_COUNT_N3, &n3Prog);
        n3Prog.finish("→ " + std::to_string(static_cast<int>(n3Result.finalAccuracy * 100)) + "% (ens:" + 
                      std::to_string(static_cast<int>(n3Result.ensembleAccuracy * 100)) + "%)");
        n3Metrics.push_back(n3Result);
        
        ProgressBar n9Prog(MAX_EPOCHS, "N=9 (11% elim)   ", 25);
        auto n9Result = trainWithEnsembleDisagreement(trainData, testData, ELIMINATION_COUNT_N9, &n9Prog);
        n9Prog.finish("→ " + std::to_string(static_cast<int>(n9Result.finalAccuracy * 100)) + "% (ens:" + 
                      std::to_string(static_cast<int>(n9Result.ensembleAccuracy * 100)) + "%)");
        n9Metrics.push_back(n9Result);
        
        std::cout << "  ╰───────────────────────────────────────────────────────────────────────────────╯\n\n";
    }
    
    // Averages
    auto avg = [](const std::vector<TrainingMetrics>& m) {
        TrainingMetrics a = {};
        for (const auto& x : m) {
            a.trainingTime += x.trainingTime;
            a.finalAccuracy += x.finalAccuracy;
            a.ensembleAccuracy += x.ensembleAccuracy;
            a.finalLoss += x.finalLoss;
            a.inferenceTime += x.inferenceTime;
        }
        int n = m.size();
        a.trainingTime /= n; a.finalAccuracy /= n; a.ensembleAccuracy /= n;
        a.finalLoss /= n; a.inferenceTime /= n;
        return a;
    };
    
    auto gdAvg = avg(gdMetrics), n3Avg = avg(n3Metrics), n9Avg = avg(n9Metrics);
    
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                              RESULTS: ENSEMBLE DISAGREEMENT                                            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Metric                    │ Gradient Descent │ N=3 (33% elim)   │ N=9 (11% elim)   │ Winner\n";
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
    
    std::cout << "  Best Network Acc (%)      │ " << std::setw(15) << (gdAvg.finalAccuracy * 100) << "%" 
              << " │ " << std::setw(15) << (n3Avg.finalAccuracy * 100) << "%"
              << " │ " << std::setw(15) << (n9Avg.finalAccuracy * 100) << "%" << " │ ";
    if (gdAvg.finalAccuracy >= n3Avg.finalAccuracy && gdAvg.finalAccuracy >= n9Avg.finalAccuracy)
        std::cout << "Gradient\n";
    else if (n3Avg.finalAccuracy >= n9Avg.finalAccuracy)
        std::cout << "N=3 (+" << std::setprecision(1) << (n3Avg.finalAccuracy - gdAvg.finalAccuracy) * 100 << "%)\n";
    else
        std::cout << "N=9 (+" << std::setprecision(1) << (n9Avg.finalAccuracy - gdAvg.finalAccuracy) * 100 << "%)\n";
    
    std::cout << std::setprecision(2);
    std::cout << "  Ensemble Voting Acc (%)   │ " << std::setw(15) << (gdAvg.ensembleAccuracy * 100) << "%" 
              << " │ " << std::setw(15) << (n3Avg.ensembleAccuracy * 100) << "%"
              << " │ " << std::setw(15) << (n9Avg.ensembleAccuracy * 100) << "%" << " │ Ensemble voting\n";
    
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

    std::cout << "\n  ENSEMBLE DISAGREEMENT PRINCIPLE:\n";
    std::cout << "  • N=3: Aggressive (33%) = faster consensus convergence\n";
    std::cout << "  • N=9: Conservative (11%) = maintains diversity longer\n";
    std::cout << "  • Networks that disagree with consensus are likely wrong\n";
    std::cout << "  • Voting provides robustness\n";
    
    double n3VsGd = (n3Avg.finalAccuracy - gdAvg.finalAccuracy) * 100;
    double n9VsGd = (n9Avg.finalAccuracy - gdAvg.finalAccuracy) * 100;
    
    std::cout << std::setprecision(1);
    std::cout << "\n  N=3 vs Gradient: " << (n3VsGd >= 0 ? "+" : "") << n3VsGd << "% accuracy, ";
    std::cout << (n3Avg.trainingTime / gdAvg.trainingTime) << "x time\n";
    std::cout << "  N=9 vs Gradient: " << (n9VsGd >= 0 ? "+" : "") << n9VsGd << "% accuracy, ";
    std::cout << (n9Avg.trainingTime / gdAvg.trainingTime) << "x time\n";
    
    std::cout << "\n";
    return 0;
}
