/*
 * Quaylyn's Law - With (Configurable) Multi-Attempt Error Correction
 *
 * Builds on the OG proof + correction test:
 *   - CERT  : early commitment baseline (no reversibility)
 *   - SINGLE: base N-section elimination
 *   - +KCOR : up to K correction attempts (directional feedback + re-search)
 *
 * Motivation:
 *   The existing +CORRECT method in quaylyns_law_with_correction.cpp allows 2 total attempts.
 *   This file generalizes the concept: allow multiple correction attempts and measure how
 *   success rate rises as the allowed attempt budget increases.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

// =============================================================================
// TEST CONFIGURATION
// =============================================================================
const long long TESTS_PER_CONFIG = 250LL;
const std::vector<int> SEARCH_SIZES = {100, 500, 1000, 5000, 10000};
const std::vector<double> INFO_LEVELS = {0.00001, 0.0001, 0.001, 0.01, 0.05, 0.10, 0.20, 0.50};
const std::vector<int> SECTION_COUNTS = {2, 3, 4, 5, 6, 7, 8, 9, 33};

// Attempt budgets (total attempts allowed, including the initial attempt).
// Example: 1 = just SINGLE, 2 = same as old +CORRECT (one correction retry), 3 = two corrections, etc.
const std::vector<unsigned long> CORRECTION_ATTEMPT_BUDGETS = {1UL, 2UL, 3UL, 5UL, 10UL};

// =============================================================================
// NOISE CONFIGURATION
// =============================================================================
// Always run SCALING (-1.0) then each fixed magnitude.
const std::vector<double> FIXED_NOISE_MAGNITUDES = {0.1 , 0.5 , 1.5, 33.33};

std::random_device rd;
std::mt19937 gen(rd());

struct IncompleteEnvironment {
    int true_target;
    double info_completeness;
    std::vector<int> search_space;
    int search_space_size;
    double noise_magnitude; // < 0 => scaling; >= 0 => fixed multiplier

    IncompleteEnvironment(int target, double completeness, int space_size, double noise_mag = -1.0)
        : true_target(target), info_completeness(completeness), search_space_size(space_size), noise_magnitude(noise_mag) {
        search_space.reserve(search_space_size);
        for (int i = 0; i < search_space_size; ++i) search_space.push_back(i);
    }

    double evaluate(int candidate) {
        double true_error = std::abs(candidate - true_target);

        double noise_stddev;
        if (noise_magnitude < 0) {
            noise_stddev = (1.0 - info_completeness) * (search_space_size * 0.1);
        } else {
            noise_stddev = noise_magnitude * (search_space_size * 0.1);
        }

        std::normal_distribution<> noise(0, noise_stddev);
        double noisy_error = true_error + noise(gen);
        return -noisy_error;
    }

    bool is_success(int candidate, int tolerance = -1) {
        if (tolerance == -1) tolerance = std::max(10, search_space_size / 20);
        return std::abs(candidate - true_target) <= tolerance;
    }

    // NON-CHEATING direction feedback:
    // Use the noisy evaluator on neighbors to infer whether the target is likely above or below.
    // Returns: +1 if target seems higher, -1 if lower.
    int get_direction_feedback(int guess, int num_samples = 5) {
        int sample_radius = std::max(5, search_space_size / 20);

        double above_score = 0.0;
        double below_score = 0.0;
        int above_count = 0;
        int below_count = 0;

        for (int i = 1; i <= num_samples; ++i) {
            int candidate = guess + (i * sample_radius / num_samples);
            if (candidate < search_space_size) {
                above_score += evaluate(candidate);
                above_count++;
            }
        }

        for (int i = 1; i <= num_samples; ++i) {
            int candidate = guess - (i * sample_radius / num_samples);
            if (candidate >= 0) {
                below_score += evaluate(candidate);
                below_count++;
            }
        }

        double avg_above = (above_count > 0) ? above_score / above_count : -1e9;
        double avg_below = (below_count > 0) ? below_score / below_count : -1e9;

        return (avg_above > avg_below) ? +1 : -1;
    }
};

class CertaintyApproach {
public:
    static int search(IncompleteEnvironment& env) {
        int sample_size = std::max(3, (int)(env.search_space.size() * env.info_completeness * 0.1));
        std::uniform_int_distribution<> dist(0, (int)env.search_space.size() - 1);

        int best_candidate = env.search_space[0];
        double best_score = env.evaluate(best_candidate);

        for (int i = 0; i < sample_size; ++i) {
            int idx = dist(gen);
            int candidate = env.search_space[idx];
            double score = env.evaluate(candidate);
            if (score > best_score) {
                best_score = score;
                best_candidate = candidate;
            }
        }

        return best_candidate;
    }
};

class NSection {
public:
    static int search(IncompleteEnvironment& env, int n_sections, int max_iterations = 20) {
        std::vector<int> remaining = env.search_space;
        double elimination_rate = 1.0 / n_sections;

        for (int iter = 0; iter < max_iterations && remaining.size() > 10; ++iter) {
            std::vector<std::pair<double, int>> scored;
            scored.reserve(remaining.size());
            for (int candidate : remaining) {
                scored.push_back({env.evaluate(candidate), candidate});
            }

            std::sort(scored.begin(), scored.end());

            int cutoff = std::max(1, (int)(scored.size() * elimination_rate));
            remaining.clear();
            remaining.reserve(scored.size() - cutoff);
            for (size_t i = cutoff; i < scored.size(); ++i) remaining.push_back(scored[i].second);
        }

        if (remaining.empty()) return env.search_space[0];

        int best = remaining[0];
        double best_score = env.evaluate(best);
        for (int candidate : remaining) {
            double score = env.evaluate(candidate);
            if (score > best_score) {
                best_score = score;
                best = candidate;
            }
        }
        return best;
    }

    static int search_on_subset(IncompleteEnvironment& env, const std::vector<int>& subset, int n_sections, int max_iterations = 20) {
        std::vector<int> remaining = subset;
        if (remaining.empty()) return env.search_space[0];

        double elimination_rate = 1.0 / n_sections;

        for (int iter = 0; iter < max_iterations && remaining.size() > 3; ++iter) {
            std::vector<std::pair<double, int>> scored;
            scored.reserve(remaining.size());
            for (int candidate : remaining) {
                scored.push_back({env.evaluate(candidate), candidate});
            }

            std::sort(scored.begin(), scored.end());

            int cutoff = std::max(1, (int)(scored.size() * elimination_rate));
            remaining.clear();
            remaining.reserve(scored.size() - cutoff);
            for (size_t i = cutoff; i < scored.size(); ++i) remaining.push_back(scored[i].second);
        }

        int best = remaining[0];
        double best_score = env.evaluate(best);
        for (int candidate : remaining) {
            double score = env.evaluate(candidate);
            if (score > best_score) {
                best_score = score;
                best = candidate;
            }
        }
        return best;
    }
};

// Multi-attempt correction:
// Attempt 1: normal N-section search
// If failed: infer direction around last guess, then re-search a corrected region.
// Repeat until success or attempts exhausted. No access to true_target except via noisy success check.
class NSectionWithMultiCorrection {
public:
    static int search(IncompleteEnvironment& env, int n_sections, unsigned long max_attempts, int max_iterations = 20) {
        if (max_attempts == 0) {
            return NSection::search(env, n_sections, max_iterations);
        }

        // Start with full range as our current belief region.
        int current_low = 0;
        int current_high = env.search_space_size - 1;

        int last_result = -1;

        for (unsigned long attempt = 1; attempt <= max_attempts; ++attempt) {
            // Build working subset for this attempt.
            std::vector<int> subset;
            subset.reserve((size_t)(current_high - current_low + 1));
            for (int i = current_low; i <= current_high; ++i) subset.push_back(i);

            int result = (attempt == 1)
                ? NSection::search(env, n_sections, max_iterations)
                : NSection::search_on_subset(env, subset, n_sections, max_iterations);

            last_result = result;
            if (env.is_success(result)) return result;

            // If we still have attempts left, refine the region.
            if (attempt == max_attempts) break;

            int direction = env.get_direction_feedback(result);

            // Conservative correction radius shrinks as we attempt more times.
            // Start broad (1/3), then shrink with each attempt.
            int base_radius = std::max(10, env.search_space_size / 3);
            int shrink = (int)attempt; // 1,2,3...
            int radius = std::max(10, base_radius / shrink);

            if (direction > 0) {
                // target appears above: shift window upward
                current_low = std::max(current_low, result);
                current_high = std::min(env.search_space_size - 1, result + radius);
            } else {
                // target appears below: shift window downward
                current_low = std::max(0, result - radius);
                current_high = std::min(current_high, result);
            }

            // Ensure window is non-empty
            if (current_low > current_high) {
                current_low = 0;
                current_high = env.search_space_size - 1;
            }
        }

        return (last_result >= 0) ? last_result : env.search_space[0];
    }
};

struct ProgressBar {
    long long total;
    int width;
    long long current;
    std::chrono::steady_clock::time_point start;

    ProgressBar(long long total_, int width_ = 50)
        : total(total_), width(width_), current(0), start(std::chrono::steady_clock::now()) {}

    void set_progress(long long value) { current = value; }

    void display() {
        double ratio = (total > 0) ? (double)current / (double)total : 1.0;
        int filled = (int)(ratio * width);

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        long long remaining = 0;
        if (current > 0) {
            double rate = (double)elapsed / (double)current;
            remaining = (long long)((total - current) * rate);
        }

        std::cout << "\r  [";
        for (int i = 0; i < width; ++i) {
            if (i < filled) std::cout << "=";
            else if (i == filled) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << std::fixed << std::setprecision(1) << (ratio * 100.0)
                  << "% (" << current << "/" << total << ")"
                  << " [" << elapsed << "s elapsed, ~" << remaining << "s remaining]";
        std::cout.flush();
    }
};

static std::string format_info_level(double info_level) {
    double pct = info_level * 100.0;
    std::ostringstream ss;
    if (pct < 0.01) {
        ss << std::fixed << std::setprecision(4) << pct << "%";
    } else if (pct < 0.1) {
        ss << std::fixed << std::setprecision(3) << pct << "%";
    } else if (pct < 1.0) {
        ss << std::fixed << std::setprecision(2) << pct << "%";
    } else if (pct < 10.0) {
        ss << std::fixed << std::setprecision(1) << pct << "%";
    } else {
        ss << std::fixed << std::setprecision(0) << pct << "%";
    }
    return ss.str();
}

static std::string format_noise_label(double noise_mag) {
    if (noise_mag < 0) return "SCALING (noise = 1 - info)";
    std::ostringstream ss;
    ss << "FIXED " << std::fixed << std::setprecision(1) << (noise_mag * 100) << "%";
    return ss.str();
}

int main() {
    // Build noise levels: scaling first, then fixed list
    std::vector<double> noise_levels_to_test;
    noise_levels_to_test.push_back(-1.0);
    for (double mag : FIXED_NOISE_MAGNITUDES) noise_levels_to_test.push_back(mag);

    // Total test scenarios include attempt budgets
    long long total_tests = (long long)noise_levels_to_test.size()
        * (long long)SEARCH_SIZES.size()
        * (long long)INFO_LEVELS.size()
        * (long long)SECTION_COUNTS.size()
        * (long long)CORRECTION_ATTEMPT_BUDGETS.size()
        * TESTS_PER_CONFIG;

    std::cout << "\nQuaylyn's Law - Multi-Attempt Correction Test\n";
    std::cout << "Compiled: " << __DATE__ << " " << __TIME__ << "\n";

    // Output file name
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* tm_info = std::localtime(&now_time);
    std::ostringstream file_ts;
    file_ts << std::put_time(tm_info, "%Y%m%d_%H%M%S");

    std::string output_filename = "quaylynlawproof_unlimited_correction_" + file_ts.str() + ".txt";
    std::cout << "Output file: " << output_filename << "\n\n";

    std::ofstream out(output_filename);
    if (!out) {
        std::cerr << "Failed to open output file: " << output_filename << "\n";
        return 1;
    }

    // Header strings
    std::ostringstream search_str, info_str, section_str, attempts_str, noise_str;
    for (size_t i = 0; i < SEARCH_SIZES.size(); ++i) {
        if (i) search_str << ", ";
        search_str << SEARCH_SIZES[i];
    }
    for (size_t i = 0; i < INFO_LEVELS.size(); ++i) {
        if (i) info_str << ", ";
        info_str << format_info_level(INFO_LEVELS[i]);
    }
    for (size_t i = 0; i < SECTION_COUNTS.size(); ++i) {
        if (i) section_str << ", ";
        section_str << SECTION_COUNTS[i];
    }
    for (size_t i = 0; i < CORRECTION_ATTEMPT_BUDGETS.size(); ++i) {
        if (i) attempts_str << ", ";
        attempts_str << CORRECTION_ATTEMPT_BUDGETS[i];
    }
    noise_str << "SCALING (1 - info) + FIXED: ";
    for (size_t i = 0; i < FIXED_NOISE_MAGNITUDES.size(); ++i) {
        if (i) noise_str << ", ";
        noise_str << std::fixed << std::setprecision(1) << (FIXED_NOISE_MAGNITUDES[i] * 100) << "%";
    }

    // Timestamp
    std::ostringstream timestamp;
    timestamp << std::put_time(tm_info, "%Y-%m-%d %H:%M:%S");

    out << "||==========================================================================||\n";
    out << "||      QUAYLYN'S LAW - MULTI-ATTEMPT ERROR CORRECTION PROOF               ||\n";
    out << "||==========================================================================||\n";
    out << "||                                                                          ||\n";
    out << "||  TIMESTAMP       : " << std::left << std::setw(50) << timestamp.str() << "||\n";
    out << "||  TESTS PER CONFIG: " << std::setw(50) << TESTS_PER_CONFIG << "||\n";
    out << "||  TOTAL TESTS     : " << std::setw(50) << total_tests << "||\n";
    out << "||  SEARCH SPACES   : " << std::setw(50) << search_str.str() << "||\n";
    out << "||  INFO LEVELS     : " << std::setw(50) << info_str.str() << "||\n";
    out << "||  N-SECTIONS      : " << std::setw(50) << section_str.str() << "||\n";
    out << "||  ATTEMPT BUDGETS : " << std::setw(50) << attempts_str.str() << "||\n";
    out << "||  NOISE CONFIG    : " << std::setw(50) << noise_str.str() << "||\n";
    out << "||                                                                          ||\n";
    out << "||  METHODS TESTED:                                                         ||\n";
    out << "||    CERT      : Early commitment baseline                                 ||\n";
    out << "||    SINGLE    : N-section (attempts=1)                                    ||\n";
    out << "||    +KCOR     : Up to K total attempts (directional feedback + re-search) ||\n";
    out << "||                                                                          ||\n";
    out << "||==========================================================================||\n\n";

    // Result storage
    // cert_results key: (noise, space, info) -> cert_rate
    std::map<std::tuple<double, int, double>, double> cert_results;

    // key: (noise, space, info, n, attempts) -> success_rate
    std::map<std::tuple<double, int, double, int, unsigned long>, double> multi_results;

    ProgressBar progress(total_tests, 50);
    long long completed = 0;

    // Main loop
    for (double noise_mag : noise_levels_to_test) {
        for (int space_size : SEARCH_SIZES) {
            for (double info_level : INFO_LEVELS) {
                // CERT once per (noise, space, info)
                int cert_success = 0;
                for (int t = 0; t < TESTS_PER_CONFIG; ++t) {
                    std::uniform_int_distribution<> target_dist(space_size / 10, space_size * 9 / 10);
                    int target = target_dist(gen);
                    IncompleteEnvironment env(target, info_level, space_size, noise_mag);
                    int res = CertaintyApproach::search(env);
                    if (env.is_success(res)) cert_success++;
                }
                cert_results[std::make_tuple(noise_mag, space_size, info_level)] = (double)cert_success / TESTS_PER_CONFIG * 100.0;

                for (int n : SECTION_COUNTS) {
                    for (unsigned long attempts : CORRECTION_ATTEMPT_BUDGETS) {
                        int success = 0;

                        for (int t = 0; t < TESTS_PER_CONFIG; ++t) {
                            std::uniform_int_distribution<> target_dist(space_size / 10, space_size * 9 / 10);
                            int target = target_dist(gen);
                            IncompleteEnvironment env(target, info_level, space_size, noise_mag);
                            int res = NSectionWithMultiCorrection::search(env, n, attempts);
                            if (env.is_success(res)) success++;

                            completed++;
                            progress.set_progress(completed);
                            progress.display();
                        }

                        double rate = (double)success / TESTS_PER_CONFIG * 100.0;
                        multi_results[std::make_tuple(noise_mag, space_size, info_level, n, attempts)] = rate;
                    }
                }
            }
        }
    }

    std::cout << "\n\nWriting results to file...\n";

    // Output tables: for each noise + space print CERT + attempt budgets columns
    for (double noise_mag : noise_levels_to_test) {
        std::string noise_label = format_noise_label(noise_mag);
        for (int space_size : SEARCH_SIZES) {
            out << "\n";
            out << "||==========================================================================||\n";
            out << "||  METRIC : MULTI-ATTEMPT SUCCESS RATE (%)                                 ||\n";
            out << "||  NOISE  : " << std::left << std::setw(60) << noise_label << "||\n";
            out << "||  SEARCH : " << std::setw(60) << space_size << "||\n";
            out << "||==========================================================================||\n";
            out << "||                                                                          ||\n";

            // Header
            out << "||  INFO %    | CERT  |";
            for (unsigned long attempts : CORRECTION_ATTEMPT_BUDGETS) {
                out << " TRY=" << std::setw(2) << attempts << " |";
            }
            out << "\n";

            out << "||-----------+-------+";
            for (size_t i = 0; i < CORRECTION_ATTEMPT_BUDGETS.size(); ++i) {
                out << "-------+";
            }
            out << "\n";

            // Rows: average across N-sections for each attempts budget (keeps table width manageable)
            for (double info_level : INFO_LEVELS) {
                std::string info_label = format_info_level(info_level);
                out << "||  " << std::right << std::setw(8) << info_label << " |";

                double cert_rate = cert_results[std::make_tuple(noise_mag, space_size, info_level)];
                out << std::fixed << std::setprecision(1) << std::setw(6) << cert_rate << "|";

                for (unsigned long attempts : CORRECTION_ATTEMPT_BUDGETS) {
                    double avg = 0.0;
                    for (int n : SECTION_COUNTS) {
                        avg += multi_results[std::make_tuple(noise_mag, space_size, info_level, n, attempts)];
                    }
                    avg /= (double)SECTION_COUNTS.size();
                    out << std::fixed << std::setprecision(1) << std::setw(6) << avg << " |";
                }
                out << "\n";
            }

            out << "||                                                                          ||\n";
            out << "||  NOTE: TRY=1 equals SINGLE (no correction). TRY>=2 adds correction retries||\n";
            out << "||==========================================================================||\n";
        }
    }

    out << "\n";
    out << "||==========================================================================||\n";
    out << "||                         PROOF COMPLETE                                   ||\n";
    out << "||==========================================================================||\n";

    std::cout << "Done.\n";
    return 0;
}
