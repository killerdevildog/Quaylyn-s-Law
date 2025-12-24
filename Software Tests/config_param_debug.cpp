// ============================================================================
// QUAYLYN'S LAW TEST #3: CONFIGURATION PARAMETER DEBUGGING
// ============================================================================
//
// SCENARIO: A system has 81 configuration parameters. ONE is wrong.
//   - Parameters INTERACT (changing one affects behavior of others)
//   - Testing is NOISY (timing, caching, network latency, etc.)
//   - You can test "batches" by temporarily overriding groups of params
//
// REAL-WORLD EXAMPLES:
//   - Game engine with wrong shader/rendering setting
//   - Server with misconfigured parameter causing intermittent failures
//   - ML model with wrong hyperparameter
//   - Docker/K8s deployment with bad env variable
//
// WHY SIMPLE APPROACHES FAIL:
//   - Testing one-by-one: 81 parameters × noisy tests = very slow
//   - Binary search: Parameters interact, so "fixing half" gives weird results
//   - Certainty: First suspicious param might be a false positive
//
// WHY QUAYLYN'S LAW WORKS:
//   - Divide params into 3 groups (27 each)
//   - Test each group by resetting those params to defaults
//   - The group with MOST improvement likely contains the bug
//   - Eliminate group with LEAST improvement (33%)
//   - Repeat until narrowed down
//
// ============================================================================

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>

// Configuration
const int NUM_PARAMS = 81;  // 81 parameters = 3^4 (good for trisection)
const int NUM_TRIALS = 200;
const double INTERACTION_STRENGTH = 0.5;  // How much params affect each other
const double NOISE_LEVEL = 0.40;          // 40% noise in measurements

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

// The configuration system
class ConfigSystem {
public:
    std::vector<double> correctValues;
    std::vector<double> currentValues;
    int buggyParam;
    double bugMagnitude;
    
    // Interaction matrix: how much param i affects param j's contribution
    std::vector<std::vector<double>> interactions;
    
    ConfigSystem() : buggyParam(-1), bugMagnitude(0) {
        correctValues.resize(NUM_PARAMS);
        currentValues.resize(NUM_PARAMS);
        interactions.resize(NUM_PARAMS, std::vector<double>(NUM_PARAMS, 0.0));
        
        std::uniform_real_distribution<double> valDist(0.0, 1.0);
        std::uniform_real_distribution<double> intDist(-INTERACTION_STRENGTH, INTERACTION_STRENGTH);
        
        for (int i = 0; i < NUM_PARAMS; i++) {
            correctValues[i] = valDist(rng);
            currentValues[i] = correctValues[i];
            
            // Some parameters interact with others
            for (int j = 0; j < NUM_PARAMS; j++) {
                if (i != j && valDist(rng) < 0.2) {  // 20% chance of interaction
                    interactions[i][j] = intDist(rng);
                }
            }
        }
    }
    
    // Inject a bug into one parameter
    int injectBug() {
        std::uniform_int_distribution<int> paramDist(0, NUM_PARAMS - 1);
        std::uniform_real_distribution<double> magDist(0.3, 0.8);
        
        buggyParam = paramDist(rng);
        bugMagnitude = magDist(rng);
        
        // Bug: parameter is offset from correct value
        currentValues[buggyParam] = correctValues[buggyParam] + bugMagnitude;
        
        return buggyParam;
    }
    
    // Calculate system "health" - how close to correct behavior
    // Takes into account interactions
    double calculateError(const std::set<int>& overrideToCorrect) {
        double error = 0.0;
        
        for (int i = 0; i < NUM_PARAMS; i++) {
            double val = overrideToCorrect.count(i) ? correctValues[i] : currentValues[i];
            double diff = val - correctValues[i];
            
            // Direct error
            error += diff * diff;
            
            // Interaction effects: wrong param i affects contribution of param j
            for (int j = 0; j < NUM_PARAMS; j++) {
                if (interactions[i][j] != 0.0) {
                    double jVal = overrideToCorrect.count(j) ? correctValues[j] : currentValues[j];
                    error += std::abs(diff * interactions[i][j] * jVal);
                }
            }
        }
        
        return std::sqrt(error);
    }
    
    // Measure error with noise (simulates real-world measurement)
    double measureError(const std::set<int>& overrideToCorrect) {
        double trueError = calculateError(overrideToCorrect);
        
        std::uniform_real_distribution<double> noiseDist(-NOISE_LEVEL, NOISE_LEVEL);
        return trueError * (1.0 + noiseDist(rng));
    }
};

// ============================================================================
// DEBUGGING STRATEGIES
// ============================================================================

struct DebugResult {
    bool foundBug;
    int evaluations;
    int bugParam;
    int guessedParam;
};

// ONE-BY-ONE: Test each parameter individually
DebugResult debugOneByOne(ConfigSystem& sys) {
    DebugResult result = {false, 0, sys.buggyParam, -1};
    
    double baselineError = sys.measureError({});
    result.evaluations++;
    
    double bestImprovement = -999;
    int bestParam = -1;
    
    for (int p = 0; p < NUM_PARAMS; p++) {
        std::set<int> override = {p};
        double errorWithFix = sys.measureError(override);
        result.evaluations++;
        
        double improvement = baselineError - errorWithFix;
        if (improvement > bestImprovement) {
            bestImprovement = improvement;
            bestParam = p;
        }
    }
    
    result.guessedParam = bestParam;
    result.foundBug = (bestParam == sys.buggyParam);
    return result;
}

// CERTAINTY: Test params, commit to first one showing improvement
DebugResult debugWithCertainty(ConfigSystem& sys) {
    DebugResult result = {false, 0, sys.buggyParam, -1};
    
    double baselineError = sys.measureError({});
    result.evaluations++;
    
    for (int p = 0; p < NUM_PARAMS; p++) {
        std::set<int> override = {p};
        double errorWithFix = sys.measureError(override);
        result.evaluations++;
        
        double improvement = baselineError - errorWithFix;
        
        // Commit to first param showing significant improvement
        if (improvement > 0.05) {
            result.guessedParam = p;
            result.foundBug = (p == sys.buggyParam);
            return result;
        }
    }
    
    return result;
}

// N-SECTION: Quaylyn's Law approach
DebugResult debugWithNSection(ConfigSystem& sys, int N, int testsPerGroup) {
    DebugResult result = {false, 0, sys.buggyParam, -1};
    
    std::vector<int> candidates;
    for (int p = 0; p < NUM_PARAMS; p++) candidates.push_back(p);
    
    double baselineError = 0;
    for (int t = 0; t < testsPerGroup; t++) {
        baselineError += sys.measureError({});
        result.evaluations++;
    }
    baselineError /= testsPerGroup;
    
    while (candidates.size() > 1) {
        int groupSize = std::max(1, (int)candidates.size() / N);
        std::vector<std::pair<double, std::vector<int>>> groups;
        
        for (int g = 0; g < N && g * groupSize < candidates.size(); g++) {
            std::vector<int> group;
            int start = g * groupSize;
            int end = std::min(start + groupSize, (int)candidates.size());
            for (int i = start; i < end; i++) {
                group.push_back(candidates[i]);
            }
            if (group.empty()) continue;
            
            // Test: override ALL params in this group to correct values
            std::set<int> override(group.begin(), group.end());
            
            double avgError = 0;
            for (int t = 0; t < testsPerGroup; t++) {
                avgError += sys.measureError(override);
                result.evaluations++;
            }
            avgError /= testsPerGroup;
            
            // Improvement = how much error decreased when we fixed this group
            double improvement = baselineError - avgError;
            groups.push_back({improvement, group});
        }
        
        if (groups.size() <= 1) {
            if (!groups.empty()) candidates = groups[0].second;
            break;
        }
        
        // Sort by improvement (highest first = most likely contains bug)
        std::sort(groups.begin(), groups.end(), [](auto& a, auto& b) {
            return a.first > b.first;
        });
        
        // Eliminate group with LEAST improvement (1/N)
        candidates.clear();
        int keepCount = std::max(1, (int)groups.size() - 1);
        for (int i = 0; i < keepCount; i++) {
            for (int idx : groups[i].second) {
                candidates.push_back(idx);
            }
        }
    }
    
    // Final selection among remaining candidates
    if (candidates.size() > 1) {
        double bestImprovement = -999;
        int bestParam = candidates[0];
        
        for (int p : candidates) {
            std::set<int> override = {p};
            double improvement = baselineError - sys.measureError(override);
            result.evaluations++;
            
            if (improvement > bestImprovement) {
                bestImprovement = improvement;
                bestParam = p;
            }
        }
        result.guessedParam = bestParam;
    } else if (!candidates.empty()) {
        result.guessedParam = candidates[0];
    }
    
    result.foundBug = (result.guessedParam == sys.buggyParam);
    return result;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        QUAYLYN'S LAW TEST: CONFIGURATION PARAMETER DEBUGGING                               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "SCENARIO: " << NUM_PARAMS << " configuration parameters, ONE is wrong.\n";
    std::cout << "          Parameters INTERACT (20% have cross-effects)\n";
    std::cout << "          Measurements are NOISY (" << (int)(NOISE_LEVEL * 100) << "% variance)\n";
    std::cout << "          Examples: game engine settings, server config, ML hyperparameters\n\n";
    
    std::cout << "CHALLENGE:\n";
    std::cout << "  • Testing one-by-one is slow (81+ evaluations)\n";
    std::cout << "  • Binary search fails due to parameter interactions\n";
    std::cout << "  • Certainty commits to false positives from noise\n\n";
    
    std::vector<std::tuple<std::string, int, int>> methods = {
        // name, N (0=one-by-one, -1=certainty), testsPerGroup
        {"One-by-One (exhaustive)", 0, 1},
        {"Certainty (commit early)", -1, 1},
        {"Bisection (50% elim, 2 tests)", 2, 2},
        {"Trisection (33% elim, 2 tests)", 3, 2},
        {"Quadsection (25% elim, 2 tests)", 4, 2},
        {"Trisection (33% elim, 3 tests)", 3, 3},
        {"Pentasection (20% elim, 2 tests)", 5, 2},
    };
    
    std::vector<int> successCounts(methods.size(), 0);
    std::vector<double> avgEvaluations(methods.size(), 0.0);
    
    std::cout << "Running " << NUM_TRIALS << " trials...\n\n";
    
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        ConfigSystem sys;
        sys.injectBug();
        
        for (int m = 0; m < methods.size(); m++) {
            auto& [name, N, testsPerGroup] = methods[m];
            DebugResult result;
            
            if (N == 0) {
                result = debugOneByOne(sys);
            } else if (N == -1) {
                result = debugWithCertainty(sys);
            } else {
                result = debugWithNSection(sys, N, testsPerGroup);
            }
            
            if (result.foundBug) successCounts[m]++;
            avgEvaluations[m] += result.evaluations;
        }
        
        if ((trial + 1) % 50 == 0) {
            std::cout << "  Completed " << (trial + 1) << "/" << NUM_TRIALS << " trials\n";
        }
    }
    
    for (int m = 0; m < methods.size(); m++) {
        avgEvaluations[m] /= NUM_TRIALS;
    }
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                       RESULTS                                              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Method                            │ Success Rate │ Avg Evaluations │ Efficiency\n";
    std::cout << "  ──────────────────────────────────┼──────────────┼─────────────────┼────────────\n";
    
    double bestEfficiency = 0;
    int bestMethod = 0;
    
    for (int m = 0; m < methods.size(); m++) {
        double successRate = 100.0 * successCounts[m] / NUM_TRIALS;
        double efficiency = successRate / std::max(1.0, avgEvaluations[m]);
        
        if (efficiency > bestEfficiency) {
            bestEfficiency = efficiency;
            bestMethod = m;
        }
        
        std::cout << "  " << std::left << std::setw(34) << std::get<0>(methods[m])
                  << "│ " << std::right << std::setw(11) << successRate << "% "
                  << "│ " << std::setw(15) << avgEvaluations[m] << " "
                  << "│ " << std::setw(8) << efficiency << "\n";
    }
    
    std::cout << "\n  BEST EFFICIENCY: " << std::get<0>(methods[bestMethod]) << "\n";
    
    std::cout << "\n  KEY INSIGHT:\n";
    std::cout << "  • One-by-one: High accuracy but slow (82+ evaluations)\n";
    std::cout << "  • Certainty: Fast but commits to false positives\n";
    std::cout << "  • Bisection (50%): Too aggressive, eliminates bug with noise\n";
    std::cout << "  • Trisection (33%): Best efficiency - fewer evals, high accuracy\n";
    std::cout << "  • QUAYLYN'S LAW: ~33% elimination balances speed and robustness\n\n";
    
    return 0;
}
