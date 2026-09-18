
#include <cmath>
#include <iostream>
#include <omp.h>

double calculate_pi_sequential(long long steps) {
    double step_size = 1.0 / steps;
    double sum = 0.0;

    for (long long i = 0; i < steps; i++) {
        double x = (i + 0.5) * step_size;
        sum += 4.0 / (1.0 + x * x);
    }

    return sum * step_size;
}

double calculate_pi_openmp(long long steps) {
    double step_size = 1.0 / steps;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < steps; i++) {
        double x = (i + 0.5) * step_size;
        sum += 4.0 / (1.0 + x * x);
    }

    return sum * step_size;
}

int main() {
    const long long steps = 500000000;

    double start = omp_get_wtime();
    double sequential_pi = calculate_pi_sequential(steps);
    double sequential_time = omp_get_wtime() - start;

    start = omp_get_wtime();
    double parallel_pi = calculate_pi_openmp(steps);
    double parallel_time = omp_get_wtime() - start;

    double difference = std::abs(sequential_pi - parallel_pi);
    double speedup = sequential_time / parallel_time;

    std::cout << "Threads available: " << omp_get_max_threads() << "\n\n";

    std::cout << "Sequential result: " << sequential_pi << "\n";
    std::cout << "Sequential time:   " << sequential_time << " seconds\n\n";

    std::cout << "OpenMP result:     " << parallel_pi << "\n";
    std::cout << "OpenMP time:       " << parallel_time << " seconds\n\n";

    std::cout << "Result difference: " << difference << "\n";
    std::cout << "Speedup:           " << speedup << "x\n";

    if (difference < 0.000001) {
        std::cout << "Correctness test:  PASSED\n";
    } else {
        std::cout << "Correctness test:  FAILED\n";
    }

    return 0;
}
