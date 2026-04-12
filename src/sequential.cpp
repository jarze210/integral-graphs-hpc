#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
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

void chunk_check_sekwencyjny(const std::vector<std::string>& chunk, std::vector<std::string>& wyniki) {
    int curr_size = static_cast<int>(chunk.size());

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
            wyniki.push_back(chunk[i]);
            std::cout << "[!] Znaleziono graf integralny: " << chunk[i] << std::endl;
        }
    }
}

int main() {
    const long long LIMIT_TESTU = 10000;
    const int MILESTONE_TESTU = 1000;

    std::cout << "=== TEST SEKWENCYJNY CPU + EIGEN (N=" << N << ", K=" << K << ") ===\n";

    std::string cmd = "geng -c " + std::to_string(N) + " " + std::to_string(K) + ":" + std::to_string(K) + " 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "Blad uruchomienia geng!" << std::endl;
        return 1;
    }

    std::vector<std::string> chunk;
    std::vector<std::string> wyniki;
    long long total_checked = 0;
    long long next_log = MILESTONE_TESTU;
    char buf[256];

    auto start_time = std::chrono::high_resolution_clock::now();

    while (fgets(buf, sizeof(buf), pipe) != nullptr && total_checked < LIMIT_TESTU) {
        std::string line(buf);
        if (!line.empty() && line.back() == '\n') line.pop_back();
        if (!line.empty() && line.back() == '\r') line.pop_back();

        chunk.push_back(line);

        if (static_cast<int>(chunk.size()) >= CHUNK_SIZE || (total_checked + static_cast<long long>(chunk.size())) >= LIMIT_TESTU) {
            chunk_check_sekwencyjny(chunk, wyniki);
            total_checked += chunk.size();

            if (total_checked >= next_log) {
                auto now = std::chrono::high_resolution_clock::now();
                double current_t = std::chrono::duration<double>(now - start_time).count();
                std::cout << "[SEQ] Przerobiono: " << total_checked << " | Czas: " << current_t << " s" << std::endl;
                next_log += MILESTONE_TESTU;
            }
            chunk.clear();
        }
    }

    pclose(pipe);
    return 0;
}