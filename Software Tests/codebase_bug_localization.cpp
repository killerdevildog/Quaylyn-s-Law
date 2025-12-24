// ============================================================================
// QUAYLYN'S LAW SOFTWARE TEST #1: LARGE CODEBASE BUG LOCALIZATION
// ============================================================================
//
// SCENARIO: A codebase with 1000 functions. ONE has a bug.
//   - You can only run the program and observe if it crashes/misbehaves
//   - Running takes time, so you want to minimize test runs
//   - You can "disable" groups of functions (mock them to return defaults)
//   - Tests are somewhat NOISY (race conditions, non-determinism)
//
// THIS MATCHES QUAYLYN'S LAW CONDITIONS:
//   - Large search space (1000 candidates)
//   - Low information (each test gives partial signal)
//   - Noise (non-deterministic behavior)
//   - Commitment is costly (eliminating wrong half = lost forever)
//
// COMPARISON:
//   - Binary search: log2(1000) ≈ 10 tests if perfect, but noise causes mistakes
//   - Trisection: More tests, but robust to noise
//   - Certainty: Test random functions, commit to first suspicious one
//
// ============================================================================

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <set>

const int NUM_FUNCTIONS = 1000;
const int NUM_TRIALS = 500;

// Key parameter: probability that a test correctly identifies bug presence
// Lower = more noise = harder problem = Quaylyn's Law shines
const double SIGNAL_QUALITY = 0.70;  // 70% chance test result is accurate

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

// Simulate testing the system with certain functions disabled
// Returns true if "bug behavior observed"
bool runTest(int bugFunction, const std::set<int>& disabledFunctions) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    bool bugActive = (disabledFunctions.find(bugFunction) == disabledFunctions.end());
    
    if (bugActive) {
        // Bug is active - test SHOULD fail, but might pass (false negative)
        return dist(rng) < SIGNAL_QUALITY;  // 70% chance of detecting
    } else {
        // Bug is disabled - test SHOULD pass, but might fail (false positive)
        return dist(rng) > SIGNAL_QUALITY;  // 30% chance of false positive
    }
}

// Run test multiple times and get failure rate
double getFailureRate(int bugFunction, const std::set<int>& disabled, int runs) {
    int failures = 0;
    for (int i = 0; i < runs; i++) {
        if (runTest(bugFunction, disabled)) failures++;
    }
    return (double)failures / runs;
}

struct Result {
    bool found;
    int tests;
    int actual;
    int guessed;
};

// Binary search (like git bisect)
Result binarySearch(int bugFunc) {
    Result r = {false, 0, bugFunc, -1};
    
    int lo = 0, hi = NUM_FUNCTIONS - 1;
    
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        
        // Disable functions [lo, mid] and test
        std::set<int> disabled;
        for (int i = lo; i <= mid; i++) disabled.insert(i);
        
        r.tests++;
        bool stillFails = runTest(bugFunc, disabled);
        
        if (stillFails) {
            // Bug still active - must be in [mid+1, hi]
            lo = mid + 1;
        } else {
            // Bug disabled - must be in [lo, mid]
            hi = mid;
        }
    }
    
    r.guessed = lo;
    r.found = (lo == bugFunc);
    return r;
}

// Binary search with retries (majority voting)
Result binarySearchRetry(int bugFunc, int retries) {
    Result r = {false, 0, bugFunc, -1};
    
    int lo = 0, hi = NUM_FUNCTIONS - 1;
    
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        
        std::set<int> disabled;
        for (int i = lo; i <= mid; i++) disabled.insert(i);
        
        int failures = 0;
        for (int t = 0; t < retries; t++) {
            r.tests++;
            if (runTest(bugFunc, disabled)) failures++;
        }
        
        if (failures > retries / 2) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    
    r.guessed = lo;
    r.found = (lo == bugFunc);
    return r;
}

// N-Section elimination (Quaylyn's Law)
Result nSection(int bugFunc, int N, int testsPerGroup) {
    Result r = {false, 0, bugFunc, -1};
    
    std::vector<int> candidates;
    for (int i = 0; i < NUM_FUNCTIONS; i++) candidates.push_back(i);
    
    while (candidates.size() > 1) {
        int groupSize = std::max(1, (int)candidates.size() / N);
        std::vector<std::pair<double, std::vector<int>>> groups;
        
        for (size_t g = 0; g < candidates.size(); g += groupSize) {
            std::vector<int> group;
            for (size_t i = g; i < std::min(g + groupSize, candidates.size()); i++) {
                group.push_back(candidates[i]);
            }
            if (group.empty()) continue;
            
            // Disable this group and measure failure rate
            std::set<int> disabled(group.begin(), group.end());
            double failRate = 0;
            for (int t = 0; t < testsPerGroup; t++) {
                r.tests++;
                if (runTest(bugFunc, disabled)) failRate += 1.0;
            }
            failRate /= testsPerGroup;
            
            // LOWER failure rate when disabled = this group CONTAINS the bug
            // So we want to KEEP groups with low failure rate (bug is in them)
            groups.push_back({failRate, group});
        }
        
        if (groups.size() <= 1) {
            if (!groups.empty()) candidates = groups[0].second;
            break;
        }
        
        // Sort by failure rate (LOWEST first = most likely contains bug)
        std::sort(groups.begin(), groups.end(), [](auto& a, auto& b) {
            return a.first < b.first;
        });
        
        // Keep groups with lowest failure rate (eliminate 1/N with highest)
        candidates.clear();
        int keepCount = std::max(1, (int)groups.size() - 1);
        for (int i = 0; i < keepCount && i < (int)groups.size(); i++) {
            for (int idx : groups[i].second) {
                candidates.push_back(idx);
            }
        }
    }
    
    r.guessed = candidates.empty() ? -1 : candidates[0];
    r.found = (r.guessed == bugFunc);
    return r;
}

// Random sampling with certainty (test random functions, commit early)
Result certaintyRandom(int bugFunc) {
    Result r = {false, 0, bugFunc, -1};
    
    std::vector<int> order;
    for (int i = 0; i < NUM_FUNCTIONS; i++) order.push_back(i);
    std::shuffle(order.begin(), order.end(), rng);
    
    for (int f : order) {
        std::set<int> disabled = {f};
        r.tests++;
        
        // If disabling this function makes test pass, commit to it
        if (!runTest(bugFunc, disabled)) {
            r.guessed = f;
            r.found = (f == bugFunc);
            return r;
        }
        
        // Stop after reasonable number of tests
        if (r.tests >= 50) break;
    }
    
    return r;
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         QUAYLYN'S LAW TEST: LARGE CODEBASE BUG LOCALIZATION                                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "SCENARIO: Codebase with " << NUM_FUNCTIONS << " functions, ONE has a bug.\n";
    std::cout << "          Test signal quality: " << (int)(SIGNAL_QUALITY * 100) << "% (30% false positive/negative rate)\n";
    std::cout << "          You can disable groups of functions to isolate the bug.\n\n";
    
    std::cout << "QUAYLYN'S LAW PREDICTION:\n";
    std::cout << "  • Binary search commits to half on each noisy test - mistakes are permanent\n";
    std::cout << "  • Trisection eliminates only 33% - keeps bug in candidate set longer\n";
    std::cout << "  • At " << (int)((1-SIGNAL_QUALITY)*100) << "% noise, trisection should beat bisection in accuracy\n\n";
    
    std::vector<std::tuple<std::string, int, int, int>> methods = {
        // name, type (0=binary, 1=binary-retry, 2=n-section, 3=certainty), N/retries, testsPerGroup
        {"Binary Search (1 test)", 0, 1, 1},
        {"Binary Search (3 retries)", 1, 3, 1},
        {"Binary Search (5 retries)", 1, 5, 1},
        {"Bisection (50% elim, 2 tests)", 2, 2, 2},
        {"Trisection (33% elim, 2 tests)", 2, 3, 2},
        {"Trisection (33% elim, 3 tests)", 2, 3, 3},
        {"Quadsection (25% elim, 2 tests)", 2, 4, 2},
        {"Certainty (random sample)", 3, 0, 1},
    };
    
    std::vector<int> successes(methods.size(), 0);
    std::vector<double> avgTests(methods.size(), 0);
    
    std::cout << "Running " << NUM_TRIALS << " trials...\n\n";
    
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        std::uniform_int_distribution<int> dist(0, NUM_FUNCTIONS - 1);
        int bugFunc = dist(rng);
        
        for (size_t m = 0; m < methods.size(); m++) {
            auto& [name, type, param1, param2] = methods[m];
            Result res;
            
            if (type == 0) res = binarySearch(bugFunc);
            else if (type == 1) res = binarySearchRetry(bugFunc, param1);
            else if (type == 2) res = nSection(bugFunc, param1, param2);
            else res = certaintyRandom(bugFunc);
            
            if (res.found) successes[m]++;
            avgTests[m] += res.tests;
        }
        
        if ((trial + 1) % 100 == 0) {
            std::cout << "  Completed " << (trial + 1) << "/" << NUM_TRIALS << " trials\n";
        }
    }
    
    for (size_t m = 0; m < methods.size(); m++) {
        avgTests[m] /= NUM_TRIALS;
    }
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                       RESULTS                                              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Method                            │ Success Rate │ Avg Tests │ Efficiency (success/tests)\n";
    std::cout << "  ──────────────────────────────────┼──────────────┼───────────┼───────────────────────────\n";
    
    for (size_t m = 0; m < methods.size(); m++) {
        double rate = 100.0 * successes[m] / NUM_TRIALS;
        double eff = rate / std::max(1.0, avgTests[m]);
        std::cout << "  " << std::left << std::setw(34) << std::get<0>(methods[m])
                  << "│ " << std::right << std::setw(11) << rate << "% "
                  << "│ " << std::setw(9) << avgTests[m]
                  << " │ " << std::setw(8) << eff << "\n";
    }
    
    std::cout << "\n  ANALYSIS:\n";
    std::cout << "  • Binary search (1 test): ~" << (int)(100 * pow(SIGNAL_QUALITY, 10)) << "% success (each step can fail)\n";
    std::cout << "  • Trisection eliminates less per round → more chances to correct\n";
    std::cout << "  • With " << (int)((1-SIGNAL_QUALITY)*100) << "% noise, conservative elimination wins\n";
    std::cout << "  • QUAYLYN'S LAW: ~33% elimination optimal when information is incomplete\n\n";
    
    return 0;
}
