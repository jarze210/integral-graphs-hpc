#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

const int N = 16;               // liczba wierzcholkow
const int K = 85;               // liczba krawedzi
const int BATCH_SIZE = 10000;   // rozmiar paczki grafow

void zdekoduj_chunki_cuda(const std::vector<std::string>& chunk, std::vector<double>& h_A, int aktualny_batch) {
    std::fill(h_A.begin(), h_A.begin() + aktualny_batch * N * N, 0.0);

    for (int i = 0; i < aktualny_batch; ++i) {
        const std::string& g6 = chunk[i];
        if (g6.empty()) continue;

        int globalBitCounter = 0;
        int offset = i * N * N;

        for (int col = 1; col < N; ++col) {
            for (int row = 0; row < col; ++row) {
                int charIndex = 1 + (globalBitCounter / 6);
                int bitPos = 5 - (globalBitCounter % 6);

                if (charIndex < static_cast<int>(g6.length())) {
                    char val = g6[charIndex] - 63;
                    if ((val & (1 << bitPos)) != 0) {
                        h_A[offset + col * N + row] = 1.0;
                        h_A[offset + row * N + col] = 1.0;
                    }
                }
                globalBitCounter++;
            }
        }
    }
}

__global__ void kernel_cykliczny(const double* d_A, int batch_count, unsigned char* d_flags) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= batch_count) return;

    double a[256];
    const double* src = d_A + id * 256;
    for (int t = 0; t < 256; ++t) a[t] = src[t];

    const double tol = 1e-7;
    const int max_sweeps = 30;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        bool changed = false;

        for (int p = 0; p < 16; ++p) {
            for (int q = p + 1; q < 16; ++q) {
                double apq = a[p * 16 + q];

                if (fabs(apq) > tol) {
                    changed = true;

                    double app = a[p * 16 + p];
                    double aqq = a[q * 16 + q];

                    double tau = (aqq - app) / (2.0 * apq);
                    double t = (tau >= 0.0)
                                   ? 1.0 / (tau + sqrt(1.0 + tau * tau))
                                   : -1.0 / (-tau + sqrt(1.0 + tau * tau));
                    double c = 1.0 / sqrt(1.0 + t * t);
                    double s = t * c;

                    a[p * 16 + p] = app - t * apq;
                    a[q * 16 + q] = aqq + t * apq;
                    a[p * 16 + q] = a[q * 16 + p] = 0.0;

                    for (int k = 0; k < 16; ++k) {
                        if (k != p && k != q) {
                            double akp = a[k * 16 + p];
                            double akq = a[k * 16 + q];
                            a[k * 16 + p] = a[p * 16 + k] = c * akp - s * akq;
                            a[k * 16 + q] = a[q * 16 + k] = s * akp + c * akq;
                        }
                    }
                }
            }
        }
        if (!changed) break;
    }

    const double EPS = 1e-7;
    for (int i = 0; i < 16; ++i) {
        double x = a[i * 16 + i];
        if (fabs(x - round(x)) > EPS) {
            d_flags[id] = 0;
            return;
        }
    }
    d_flags[id] = 1;
}

int main() {
    const long long LIMIT_TESTU = 1000000;
    const int MILESTONE_TESTU = 100000;

    std::vector<double> h_A(BATCH_SIZE * N * N, 0.0);
    std::vector<unsigned char> h_flags(BATCH_SIZE, 0);

    double* d_A;
    unsigned char* d_flags;
    cudaMalloc((void**)&d_A, BATCH_SIZE * N * N * sizeof(double));
    cudaMalloc((void**)&d_flags, BATCH_SIZE * sizeof(unsigned char));

    std::string cmd = "geng -c " + std::to_string(N) + " " + std::to_string(K) + ":" + std::to_string(K) + " 2>/dev/null";
    std::cout << "=== CYCLIC JACOBI CUDA (N=" << N << ", K=" << K << ") ===\n";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "Blad uruchomienia geng!" << std::endl;
        return 1;
    }

    std::vector<std::string> chunk;
    char buf[256];
    long long total_checked = 0;
    long long next_log = MILESTONE_TESTU;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (fgets(buf, sizeof(buf), pipe) != nullptr && total_checked < LIMIT_TESTU) {
        std::string line(buf);
        if (!line.empty() && line.back() == '\n') line.pop_back();
        if (!line.empty() && line.back() == '\r') line.pop_back();

        chunk.push_back(line);

        if (static_cast<int>(chunk.size()) >= BATCH_SIZE || (total_checked + static_cast<long long>(chunk.size())) >= LIMIT_TESTU) {
            int current_batch = static_cast<int>(chunk.size());
            zdekoduj_chunki_cuda(chunk, h_A, current_batch);

            cudaMemcpy(d_A, h_A.data(), current_batch * N * N * sizeof(double), cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (current_batch + threads - 1) / threads;
            kernel_cykliczny<<<blocks, threads>>>(d_A, current_batch, d_flags);

            cudaMemcpy(h_flags.data(), d_flags, current_batch * sizeof(unsigned char), cudaMemcpyDeviceToHost);

            for (int i = 0; i < current_batch; ++i) {
                if (h_flags[i] == 1) {
                    std::cout << "[!] Znaleziono graf integralny: " << chunk[i] << std::endl;
                }
            }

            total_checked += current_batch;

            if (total_checked >= next_log) {
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms = 0;
                cudaEventElapsedTime(&ms, start, stop);
                std::cout << "[GPU Cyclic] Przerobiono: " << total_checked
                          << " | Czas: " << ms / 1000.0 << " s" << std::endl;
                next_log += MILESTONE_TESTU;
            }
            chunk.clear();
        }
    }

    pclose(pipe);
    cudaFree(d_A);
    cudaFree(d_flags);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}