#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <immintrin.h>

const int N = 16;                // liczba wierzcholkow
const int K = 85;                // liczba krawedzi
const double TOLERANCJA = 1e-7;  // tolerancja dla wartosci wlasnych
const int CHUNK_SIZE = 1000;     // rozmiar paczki grafow

void graph6_to_adjMatrix(const std::string& graph6String, double* a) {
    std::memset(a, 0, 256 * sizeof(double));
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
                    a[row * N + col] = 1.0;
                    a[col * N + row] = 1.0;
                }
            }
            globalBitCounter++;
        }
    }
}

void check_chunk_avx(const std::vector<std::string>& chunk, bool use_omp) {
    int curr_size = static_cast<int>(chunk.size());

    #pragma omp parallel for if(use_omp) schedule(dynamic, 100)
    for (int i = 0; i < curr_size; ++i) {
        // Wyrownanie pamieci do 32 bajtow pod instrukcje AVX2
        alignas(32) double a[256];
        graph6_to_adjMatrix(chunk[i], a);

        const int max_sweeps = 30;
        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            bool changed = false;

            for (int p = 0; p < 16; ++p) {
                for (int q = p + 1; q < 16; ++q) {
                    double apq = a[p * 16 + q];

                    if (std::abs(apq) > TOLERANCJA) {
                        changed = true;

                        double app = a[p * 16 + p];
                        double aqq = a[q * 16 + q];

                        double tau = (aqq - app) / (2.0 * apq);
                        double t = (tau >= 0.0)
                                       ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
                                       : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
                        double c = 1.0 / std::sqrt(1.0 + t * t);
                        double s = t * c;

                        // Wektoryzacja AVX2 + FMA dla wierszy p i q
                        __m256d vc = _mm256_set1_pd(c);
                        __m256d vs = _mm256_set1_pd(s);

                        for (int v = 0; v < 4; ++v) {
                            __m256d ap_vec = _mm256_load_pd(&a[p * 16 + v * 4]);
                            __m256d aq_vec = _mm256_load_pd(&a[q * 16 + v * 4]);

                            __m256d ap_new = _mm256_fmsub_pd(vc, ap_vec, _mm256_mul_pd(vs, aq_vec));
                            __m256d aq_new = _mm256_fmadd_pd(vs, ap_vec, _mm256_mul_pd(vc, aq_vec));

                            _mm256_store_pd(&a[p * 16 + v * 4], ap_new);
                            _mm256_store_pd(&a[q * 16 + v * 4], aq_new);
                        }

                        // Utrzymanie symetrii kolumn oraz aktualizacja przekatnej
                        for (int k = 0; k < 16; ++k) {
                            a[k * 16 + p] = a[p * 16 + k];
                            a[k * 16 + q] = a[q * 16 + k];
                        }
                        a[p * 16 + p] = app - t * apq;
                        a[q * 16 + q] = aqq + t * apq;
                        a[p * 16 + q] = 0.0;
                        a[q * 16 + p] = 0.0;
                    }
                }
            }
            if (!changed) break;
        }

        // Weryfikacja integralnosci
        bool isIntegral = true;
        for (int j = 0; j < 16; ++j) {
            double val = a[j * 16 + j];
            if (std::abs(val - std::round(val)) > TOLERANCJA) {
                isIntegral = false;
                break;
            }
        }

        if (isIntegral) {
            #pragma omp critical
            {
                std::cout << "[!] Znaleziono graf integralny: " << chunk[i] << std::endl;
            }
        }
    }
}

int main() {
    const long long LIMIT_TESTU = 100000;
    const int MILESTONE_TESTU = 10000;
    omp_set_num_threads(16);

    std::cout << "=== TEST OPENMP + AVX2 + FMA (16T, N=" << N << ", K=" << K << ") ===\n";

    std::string cmd = "geng -c " + std::to_string(N) + " " + std::to_string(K) + ":" + std::to_string(K) + " 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "Blad uruchomienia geng!" << std::endl;
        return 1;
    }

    std::vector<std::string> chunk;
    long long total_checked = 0;
    long long next_log = MILESTONE_TESTU;
    char buf[256];

    double start_time = omp_get_wtime();

    while (fgets(buf, sizeof(buf), pipe) != nullptr && total_checked < LIMIT_TESTU) {
        std::string line(buf);
        if (!line.empty() && line.back() == '\n') line.pop_back();
        if (!line.empty() && line.back() == '\r') line.pop_back();

        chunk.push_back(line);

        if (static_cast<int>(chunk.size()) >= CHUNK_SIZE || (total_checked + static_cast<long long>(chunk.size())) >= LIMIT_TESTU) {
            check_chunk_avx(chunk, true);
            total_checked += chunk.size();

            if (total_checked >= next_log) {
                double current_t = omp_get_wtime() - start_time;
                std::cout << "[AVX2] Przerobiono: " << total_checked << " | Czas: " << current_t << " s" << std::endl;
                next_log += MILESTONE_TESTU;
            }
            chunk.clear();
        }
    }

    pclose(pipe);
    return 0;
}