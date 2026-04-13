#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <Eigen/Dense>

const int N = 16;                // liczba wierzcholkow
const int K = 85;                // liczba krawedzi
const double TOLERANCJA = 1e-7;  // tolerancja dla wartosci wlasnych
const int CHUNK_SIZE = 1000;     // rozmiar paczki grafow

typedef Eigen::Matrix<double, N, N> Matrix16;

// Dekodowanie formatu graph6 do macierzy sasiedztwa (tylko dla N=16)
void graph6_to_adjMatrix(const std::string& graph6String, Matrix16& adjacencyMatrix) {
    adjacencyMatrix.setZero();
    if (graph6String.empty()) return;

    const int dataOffset = 1;
    int globalBitCounter = 0;

    for (int col = 1; col < N; ++col) {
        for (int row = 0; row < col; ++row) {
            int charIndex = dataOffset + (globalBitCounter / 6);
            int bitPositionInChar = 5 - (globalBitCounter % 6);

            if (charIndex < static_cast<int>(graph6String.length())) {
                char decodedCharValue = graph6String[charIndex] - 63;
                if ((decodedCharValue & (1 << bitPositionInChar)) != 0) {
                    adjacencyMatrix(row, col) = 1.0;
                    adjacencyMatrix(col, row) = 1.0;
                }
            }
            globalBitCounter++;
        }
    }
}

void chunk_check(const std::vector<std::string>& chunk, std::vector<std::string>& wyniki) {
    int curr_size = static_cast<int>(chunk.size());

    #pragma omp parallel for schedule(dynamic, 100)
    for (int i = 0; i < curr_size; ++i) {
        Matrix16 adj;
        graph6_to_adjMatrix(chunk[i], adj);

        Eigen::SelfAdjointEigenSolver<Matrix16> solver(adj, Eigen::EigenvaluesOnly);
        auto eigenvalues = solver.eigenvalues();

        bool isIntegral = true;
        for (int j = 0; j < eigenvalues.size(); ++j) {
            double val = eigenvalues(j);
            if (std::abs(val - std::round(val)) > TOLERANCJA) {
                isIntegral = false;
                break;
            }
        }

        if (isIntegral) {
            #pragma omp critical
            {
                wyniki.push_back(chunk[i]);
                std::cout << "[!] Znaleziono graf integralny: " << chunk[i] << std::endl;
            }
        }
    }
}

int main() {
    std::vector<int> test_threads = {1, 2, 4, 8, 12, 16};
    const long long LIMIT_TESTU = 10000;
    const int MILESTONE_TESTU = 1000;

    std::cout << "=== TEST WYDAJNOSCI OPENMP (N=" << N << ", K=" << K << ") ===\n";

    for (int num_t : test_threads) {
        omp_set_num_threads(num_t);
        std::string cmd = "geng -c " + std::to_string(N) + " " + std::to_string(K) + ":" + std::to_string(K) + " 2>/dev/null";

        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            std::cerr << "Blad uruchomienia geng!" << std::endl;
            return 1;
        }

        std::vector<std::string> chunk;
        std::vector<std::string> wyniki_temp;
        long long total_checked = 0;
        long long next_log = MILESTONE_TESTU;
        char buf[256];

        std::cout << ">>> START: " << num_t << " watkow" << std::endl;
        double start_time = omp_get_wtime();

        while (fgets(buf, sizeof(buf), pipe) != nullptr && total_checked < LIMIT_TESTU) {
            std::string line(buf);
            if (!line.empty() && line.back() == '\n') line.pop_back();
            if (!line.empty() && line.back() == '\r') line.pop_back();

            chunk.push_back(line);

            if (static_cast<int>(chunk.size()) >= CHUNK_SIZE || (total_checked + static_cast<long long>(chunk.size())) >= LIMIT_TESTU) {
                chunk_check(chunk, wyniki_temp);
                total_checked += chunk.size();

                if (total_checked >= next_log) {
                    double current_t = omp_get_wtime() - start_time;
                    std::cout << "[" << num_t << "T] Przerobiono: " << total_checked
                              << " | Czas: " << current_t << " s" << std::endl;
                    next_log += MILESTONE_TESTU;
                }
                chunk.clear();
            }
        }

        pclose(pipe);
    }

    return 0;
}