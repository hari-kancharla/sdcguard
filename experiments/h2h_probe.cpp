// Ground truth for the head-to-head test (scripts/modal_h2h.py): makes the same cuBLAS calls DCGM's diagnostic
// plugin and gpu-burn make, plus the FP16/BF16 calls known to hit cuBLAS known issue 6862950, and says for each
// whether the result was right. Host code only (no nvcc needed):
//
//   g++ -O2 -std=c++17 h2h_probe.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublasLt -lcublas
//       -lcudart -o probe
//   ./probe hgemm:2048 sgemm:2048 sgemmtc:2048 sgemm:8192 sgemmtc:8192 dgemm:8192 ex16:2048 exbf16:2048 scrub
//   ./probe lt16:1000:2048:16 ltbf16:2048:2048:2048          (preflight: every cuBLASLt algorithm, see below)
//
// Each case prints one JSON line. The output buffer is filled with 0xFF bytes first (NaN in FP16, BF16, FP32 and
// FP64), so elements a GEMM never wrote are counted exactly ("unwritten"). Sampled columns are compared with a
// float64 CPU reference computed from the same (rounded) inputs, with the loose tolerance the public reproduction
// uses: |c - ref| <= 5e-2 * (1 + |ref|). Inputs follow DCGM and gpu-burn: srand(10), rand() % 1000000 / 100000.0.
//
//   hgemm    cublasHgemm, FP16 in/out/compute          (DCGM diagnostic, half precision)
//   sgemm    cublasSgemm, default math                 (DCGM single precision; gpu-burn default)
//   sgemmtc  cublasSgemm with CUBLAS_TENSOR_OP_MATH    (DCGM after halfway; gpu-burn -tc)
//   dgemm    cublasDgemm                               (DCGM double; gpu-burn -d)
//   ex16     cublasGemmEx FP16 in/out, FP32 compute    (what PyTorch fp16 matmul uses; affected per the repro)
//   exbf16   cublasGemmEx BF16 in/out, FP32 compute    (what PyTorch bf16 matmul uses; affected per the repro)
//   scrub    is freshly allocated device memory zero? (decides whether self-comparison tools can see holes)
//   lt16 / ltbf16 M:K:N   cublasLtMatmul FP16 / BF16, FP32 compute, laid out the way PyTorch's linear layer calls
//            it (op(A) = A^T, M outputs per token, N tokens). Runs every algorithm the heuristic offers (up to 16),
//            each into a NaN-filled buffer, and prints its CUBLASLT_ALGO_CONFIG_ID and cluster shape with the result;
//            then the first one again with its cluster forced to 1x1x1 (the documented workaround). Known issue
//            6862950 says only algorithm 66 with 4 or more CTAs per cluster is affected: this shows whether that
//            is what goes wrong on this machine.
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint16_t f2h(float f) {                 // float -> IEEE half, round to nearest even
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t h = mant >> shift, rem = mant & ((1u << shift) - 1u), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)exp << 10) | (mant >> 13), rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return (uint16_t)(sign | h);
}

static float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FFu, x;
    if (exp == 0) {
        if (mant == 0) {
            x = sign;
        } else {
            int e = -1;
            do { mant <<= 1; e++; } while (!(mant & 0x400u));
            x = sign | ((uint32_t)(127 - 15 - e) << 23) | ((mant & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7F800000u | (mant << 13);
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

static uint16_t f2bf(float f) {                // float -> bfloat16, round to nearest even
    uint32_t x;
    std::memcpy(&x, &f, 4);
    if ((x & 0x7F800000u) == 0x7F800000u) return (uint16_t)((x >> 16) | ((x & 0xFFFFu) ? 0x40u : 0u));
    x += 0x7FFFu + ((x >> 16) & 1u);
    return (uint16_t)(x >> 16);
}

static float bf2f(uint16_t b) {
    uint32_t x = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

static void fail(const char *name, int n, const char *what, int code) {
    std::printf("{\"case\": \"%s\", \"n\": %d, \"error\": \"%s\", \"code\": %d}\n", name, n, what, code);
    std::fflush(stdout);
}

#define CUDA_OR(call, what)                                                       \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) { fail(name, n, what, (int)e_); return; }          \
    } while (0)
#define CUBLAS_OR(call, what)                                                     \
    do {                                                                          \
        cublasStatus_t s_ = (call);                                               \
        if (s_ != CUBLAS_STATUS_SUCCESS) { fail(name, n, what, (int)s_); return; } \
    } while (0)

static void scrub() {
    const char *name = "scrub";
    int n = 0;
    size_t free_b = 0, total_b = 0;
    CUDA_OR(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    size_t want = free_b / 2, sample = (size_t)256 << 20;
    if (want < sample) want = sample;
    void *p = nullptr;
    CUDA_OR(cudaMalloc(&p, want), "cudaMalloc");
    std::vector<unsigned char> h(sample);
    size_t nonzero = 0, offsets = 4;
    for (size_t o = 0; o < offsets; o++) {      // four evenly spaced 64 MiB windows of the allocation
        size_t chunk = sample / offsets, off = (want - chunk) / (offsets - 1) * o;
        CUDA_OR(cudaMemcpy(h.data(), (unsigned char *)p + off, chunk, cudaMemcpyDeviceToHost), "cudaMemcpy");
        for (size_t i = 0; i < chunk; i++) nonzero += h[i] != 0;
    }
    cudaFree(p);
    std::printf("{\"case\": \"scrub\", \"allocated_bytes\": %zu, \"sampled_bytes\": %zu, \"nonzero_bytes\": %zu, "
                "\"fresh_memory_zeroed\": %s}\n", want, sample, nonzero, nonzero == 0 ? "true" : "false");
    std::fflush(stdout);
}

static int elem_size(const std::string &kind) {
    return (kind == "dgemm") ? 8 : (kind == "sgemm" || kind == "sgemmtc") ? 4 : 2;
}

// inputs as DCGM and gpu-burn make them; A and B come back rounded to the type actually given to cuBLAS
static void make_inputs(const std::string &kind, int n, std::vector<double> &A, std::vector<double> &B,
                        std::vector<unsigned char> &hA, std::vector<unsigned char> &hB) {
    const size_t nn = (size_t)n * n;
    const int esize = elem_size(kind);
    A.assign(nn, 0.0);
    B.assign(nn, 0.0);
    hA.assign(nn * esize, 0);
    hB.assign(nn * esize, 0);
    std::srand(10);
    for (size_t i = 0; i < nn; i++) A[i] = (double)(std::rand() % 1000000) / 100000.0;
    for (size_t i = 0; i < nn; i++) B[i] = (double)(std::rand() % 1000000) / 100000.0;
    for (size_t i = 0; i < nn; i++) {
        for (int w = 0; w < 2; w++) {
            double &v = w ? B[i] : A[i];
            unsigned char *dst = (w ? hB.data() : hA.data()) + i * esize;
            if (esize == 8) {
                std::memcpy(dst, &v, 8);
            } else if (esize == 4) {
                float f = (float)v;
                std::memcpy(dst, &f, 4);
                v = f;
            } else if (kind == "exbf16" || kind == "ltbf16") {
                uint16_t b = f2bf((float)v);
                std::memcpy(dst, &b, 2);
                v = bf2f(b);
            } else {
                uint16_t b = f2h((float)v);
                std::memcpy(dst, &b, 2);
                v = h2f(b);
            }
        }
    }
}

static double decode(const std::string &kind, const unsigned char *p) {
    const int esize = elem_size(kind);
    if (esize == 8) { double d; std::memcpy(&d, p, 8); return d; }
    if (esize == 4) { float f; std::memcpy(&f, p, 4); return f; }
    uint16_t b;
    std::memcpy(&b, p, 2);
    return (kind == "exbf16" || kind == "ltbf16") ? bf2f(b) : h2f(b);
}

static void encode(const std::string &kind, double v, unsigned char *p) {
    const int esize = elem_size(kind);
    if (esize == 8) { std::memcpy(p, &v, 8); return; }
    if (esize == 4) { float f = (float)v; std::memcpy(p, &f, 4); return; }
    uint16_t b = (kind == "exbf16" || kind == "ltbf16") ? f2bf((float)v) : f2h((float)v);
    std::memcpy(p, &b, 2);
}

// counts unwritten (all 0xFF) and non-finite elements, compares 32 evenly spaced columns with float64, prints JSON
static void account(const std::string &kind, int n, const std::vector<double> &A, const std::vector<double> &B,
                    const std::vector<unsigned char> &hC, int ver, float ms, const char *source) {
    const char *name = kind.c_str();
    const size_t nn = (size_t)n * n;
    const int esize = elem_size(kind);
    auto value = [&](size_t i) -> double { return decode(kind, hC.data() + i * esize); };
    size_t unwritten = 0, nonfinite = 0;
    long long first_unwritten = -1;
    for (size_t i = 0; i < nn; i++) {
        const unsigned char *p = hC.data() + i * esize;
        bool all_ff = true;
        for (int b = 0; b < esize; b++) all_ff = all_ff && p[b] == 0xFF;
        if (all_ff) {
            unwritten++;
            if (first_unwritten < 0) first_unwritten = (long long)i;
        } else if (!std::isfinite(value(i))) {
            nonfinite++;
        }
    }
    // float64 reference on 32 evenly spaced columns (column-major: C[i + j n] = sum_k A[i + k n] B[k + j n])
    const int cols = n < 32 ? n : 32;
    size_t sampled = 0, wrong = 0;
    double worst = 0.0;
    std::vector<double> ref(n);
    for (int c = 0; c < cols; c++) {
        size_t j = cols == 1 ? 0 : (size_t)((long long)c * (n - 1) / (cols - 1));
        std::fill(ref.begin(), ref.end(), 0.0);
        for (size_t k = 0; k < (size_t)n; k++) {
            double b = B[k + j * n];
            const double *a = &A[k * n];
            for (size_t i = 0; i < (size_t)n; i++) ref[i] += a[i] * b;
        }
        for (size_t i = 0; i < (size_t)n; i++) {
            double got = value(i + j * n), r = ref[i], err = std::fabs(got - r) / (1.0 + std::fabs(r));
            sampled++;
            if (!std::isfinite(got) || err > 5e-2) wrong++;
            if (std::isfinite(err) && err > worst) worst = err;
        }
    }
    std::printf("{\"case\": \"%s\", \"n\": %d, \"source\": \"%s\", \"cublas_version\": %d, \"elements\": %zu, "
                "\"unwritten\": %zu, \"first_unwritten_index\": %lld, \"nonfinite_written\": %zu, \"sampled\": %zu, "
                "\"wrong_sampled\": %zu, \"worst_rel_err_finite\": %.3g, \"gemm_ms\": %.3f, \"correct\": %s}\n",
                name, n, source, ver, nn, unwritten, first_unwritten, nonfinite, sampled, wrong, worst, ms,
                (unwritten == 0 && nonfinite == 0 && wrong == 0) ? "true" : "false");
    std::fflush(stdout);
}

// CPU-only check of the accounting above: C computed on the CPU in the case's type, optionally with a hole
static void cpu_case(const std::string &kind, int n, bool hole) {
    std::vector<double> A, B;
    std::vector<unsigned char> hA, hB;
    make_inputs(kind, n, A, B, hA, hB);
    const int esize = elem_size(kind);
    std::vector<unsigned char> hC((size_t)n * n * esize, 0xFF);
    for (size_t j = 0; j < (size_t)n; j++)
        for (size_t i = 0; i < (size_t)n; i++) {
            if (hole && j >= (size_t)n / 4 && j < (size_t)n / 2) continue;      // a quarter of the columns unwritten
            double s = 0.0;
            for (size_t k = 0; k < (size_t)n; k++) s += A[i + k * n] * B[k + j * n];
            encode(kind, s, hC.data() + (i + j * n) * esize);
        }
    account(kind, n, A, B, hC, 0, 0.0f, hole ? "cpu_with_hole" : "cpu");
}

static void gemm_case(const std::string &kind, int n) {
    const char *name = kind.c_str();
    const size_t nn = (size_t)n * n;
    const int esize = elem_size(kind);
    std::vector<double> A, B;
    std::vector<unsigned char> hA, hB;
    make_inputs(kind, n, A, B, hA, hB);
    void *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_OR(cudaMalloc(&dA, nn * esize), "cudaMalloc A");
    CUDA_OR(cudaMalloc(&dB, nn * esize), "cudaMalloc B");
    CUDA_OR(cudaMalloc(&dC, nn * esize), "cudaMalloc C");
    CUDA_OR(cudaMemcpy(dA, hA.data(), nn * esize, cudaMemcpyHostToDevice), "copy A");
    CUDA_OR(cudaMemcpy(dB, hB.data(), nn * esize, cudaMemcpyHostToDevice), "copy B");
    CUDA_OR(cudaMemset(dC, 0xFF, nn * esize), "memset C");
    cublasHandle_t h = nullptr;
    CUBLAS_OR(cublasCreate(&h), "cublasCreate");
    int ver = 0;
    cublasGetVersion(h, &ver);
    if (kind == "sgemmtc") CUBLAS_OR(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    if (kind == "hgemm") {
        __half_raw o, z;
        o.x = f2h(1.0f);
        z.x = f2h(0.0f);
        __half one(o), zero(z);
        CUBLAS_OR(cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, (const __half *)dA, n, (const __half *)dB, n,
                              &zero, (__half *)dC, n), "cublasHgemm");
    } else if (kind == "sgemm" || kind == "sgemmtc") {
        float one = 1.0f, zero = 0.0f;
        CUBLAS_OR(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, (const float *)dA, n, (const float *)dB, n,
                              &zero, (float *)dC, n), "cublasSgemm");
    } else if (kind == "dgemm") {
        double one = 1.0, zero = 0.0;
        CUBLAS_OR(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, (const double *)dA, n, (const double *)dB, n,
                              &zero, (double *)dC, n), "cublasDgemm");
    } else {
        float one = 1.0f, zero = 0.0f;
        cudaDataType_t t = kind == "exbf16" ? CUDA_R_16BF : CUDA_R_16F;
        CUBLAS_OR(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, dA, t, n, dB, t, n, &zero, dC, t, n,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "cublasGemmEx");
    }
    cudaEventRecord(t1);
    CUDA_OR(cudaDeviceSynchronize(), "sync after gemm");
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    std::vector<unsigned char> hC(nn * esize);
    CUDA_OR(cudaMemcpy(hC.data(), dC, nn * esize, cudaMemcpyDeviceToHost), "copy C");
    cublasDestroy(h);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    account(kind, n, A, B, hC, ver, ms, "gpu");
}

// ---------------------------------------------------------------- cuBLASLt: which algorithms go wrong
static int cluster_ctas(int id) {               // CTAs per cluster for a cublasLtClusterShape_t (0: auto/unknown)
    static const int x[] = {0, 0, 1, 2, 4, 1, 2, 4, 1, 2, 4, 8, 1, 8, 2, 16, 1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15,
                            3, 5, 6, 7, 1, 2, 3, 4, 5, 3, 1, 2, 3, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1};
    static const int y[] = {0, 0, 1, 1, 1, 2, 2, 2, 4, 4, 4, 1, 8, 2, 8, 1, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 5, 6, 6, 7, 7, 9, 10, 11, 12, 13, 14, 15};
    if (id < 2 || id >= (int)(sizeof(x) / sizeof(x[0]))) return 0;
    return x[id] * y[id];
}

static void lt_case(const std::string &kind, int M, int K, int N) {
    char label[96];
    std::snprintf(label, sizeof label, "%dx%dx%d", M, K, N);
    const char *name = kind.c_str();
    int n = 0;
    const bool bf = kind == "ltbf16";
    const cudaDataType_t t = bf ? CUDA_R_16BF : CUDA_R_16F;
    // A: K x M (column-major, used transposed), B: K x N, D: M x N
    std::vector<double> A((size_t)K * M), B((size_t)K * N);
    std::vector<uint16_t> hA(A.size()), hB(B.size());
    std::srand(12);
    for (size_t i = 0; i < A.size(); i++) {
        float f = (float)((std::rand() % 2001) - 1000) / 2000.0f;
        hA[i] = bf ? f2bf(f) : f2h(f);
        A[i] = bf ? bf2f(hA[i]) : h2f(hA[i]);
    }
    for (size_t i = 0; i < B.size(); i++) {
        float f = (float)((std::rand() % 2001) - 1000) / 2000.0f;
        hB[i] = bf ? f2bf(f) : f2h(f);
        B[i] = bf ? bf2f(hB[i]) : h2f(hB[i]);
    }
    const int cols = N < 32 ? N : 32;            // float64 reference on up to 32 evenly spaced tokens
    std::vector<int> js(cols);
    std::vector<double> ref((size_t)M * cols, 0.0);
    for (int c = 0; c < cols; c++) {
        js[c] = cols == 1 ? 0 : (int)((long long)c * (N - 1) / (cols - 1));
        for (int i = 0; i < M; i++) {
            double s = 0.0;
            for (int kk = 0; kk < K; kk++) s += A[kk + (size_t)i * K] * B[kk + (size_t)js[c] * K];
            ref[(size_t)i + (size_t)c * M] = s;
        }
    }
    void *dA = nullptr, *dB = nullptr, *dD = nullptr, *ws = nullptr;
    const size_t wsize = (size_t)32 << 20;
    CUDA_OR(cudaMalloc(&dA, A.size() * 2), "cudaMalloc A");
    CUDA_OR(cudaMalloc(&dB, B.size() * 2), "cudaMalloc B");
    CUDA_OR(cudaMalloc(&dD, (size_t)M * N * 2), "cudaMalloc D");
    CUDA_OR(cudaMalloc(&ws, wsize), "cudaMalloc workspace");
    CUDA_OR(cudaMemcpy(dA, hA.data(), A.size() * 2, cudaMemcpyHostToDevice), "copy A");
    CUDA_OR(cudaMemcpy(dB, hB.data(), B.size() * 2, cudaMemcpyHostToDevice), "copy B");
    cublasLtHandle_t lt = nullptr;
    CUBLAS_OR(cublasLtCreate(&lt), "cublasLtCreate");
    cublasLtMatmulDesc_t op = nullptr;
    CUBLAS_OR(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CUBLAS_OR(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta), "set transa");
    CUBLAS_OR(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb), "set transb");
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
    CUBLAS_OR(cublasLtMatrixLayoutCreate(&la, t, K, M, K), "layout A");
    CUBLAS_OR(cublasLtMatrixLayoutCreate(&lb, t, K, N, K), "layout B");
    CUBLAS_OR(cublasLtMatrixLayoutCreate(&ld, t, M, N, M), "layout D");
    cublasLtMatmulPreference_t pref = nullptr;
    CUBLAS_OR(cublasLtMatmulPreferenceCreate(&pref), "preference");
    CUBLAS_OR(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsize,
                                                   sizeof wsize), "set workspace");
    cublasLtMatmulHeuristicResult_t heur[16] = {};
    int found = 0;
    CUBLAS_OR(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 16, heur, &found), "heuristic");
    int ver = (int)cublasLtGetVersion();
    int sms = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    std::vector<uint16_t> hD((size_t)M * N);
    auto run = [&](const cublasLtMatmulAlgo_t &algo, int idx, const char *variant) {
        int32_t algo_id = -1, splitk = 0;
        uint16_t cluster = 0;
        uint32_t tile = 0, stages = 0;
        size_t got = 0;
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof algo_id, &got);
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster, sizeof cluster,
                                             &got);
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, &got);
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages, &got);
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof splitk, &got);
        cudaMemset(dD, 0xFF, (size_t)M * N * 2);
        float one = 1.0f, zero = 0.0f;
        cublasStatus_t st = cublasLtMatmul(lt, op, &one, dA, la, dB, lb, &zero, dD, ld, dD, ld, &algo, ws, wsize, 0);
        cudaError_t ce = cudaDeviceSynchronize();
        size_t unwritten = 0, wrong = 0, sampled = 0;
        double worst = 0.0;
        if (st == CUBLAS_STATUS_SUCCESS && ce == cudaSuccess) {
            cudaMemcpy(hD.data(), dD, hD.size() * 2, cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < hD.size(); i++) unwritten += hD[i] == 0xFFFF;
            for (int c = 0; c < cols; c++)
                for (int i = 0; i < M; i++) {
                    uint16_t v = hD[(size_t)i + (size_t)js[c] * M];
                    double g = bf ? bf2f(v) : h2f(v), r = ref[(size_t)i + (size_t)c * M];
                    double e = std::fabs(g - r) / (1.0 + std::fabs(r));
                    sampled++;
                    if (!std::isfinite(g) || e > 5e-2) wrong++;
                    if (std::isfinite(e) && e > worst) worst = e;
                }
        }
        std::printf("{\"case\": \"%s\", \"n\": \"%s#%d%s\", \"variant\": \"%s\", \"cublaslt_version\": %d, "
                    "\"sms_visible\": %d, \"algo_id\": %d, \"cluster_shape_id\": %u, \"cluster_ctas\": %d, "
                    "\"tile_id\": %u, \"stages_id\": %u, \"splitk\": %d, \"status\": %d, \"cuda_error\": %d, "
                    "\"elements\": %zu, \"unwritten\": %zu, \"sampled\": %zu, \"wrong_sampled\": %zu, "
                    "\"worst_rel_err_finite\": %.3g, \"correct\": %s}\n",
                    name, label, idx, variant[0] ? "w" : "", variant, ver, sms, algo_id, (unsigned)cluster,
                    cluster_ctas(cluster), tile, stages, splitk, (int)st, (int)ce, hD.size(), unwritten, sampled,
                    wrong, worst,
                    (st == CUBLAS_STATUS_SUCCESS && ce == cudaSuccess && unwritten == 0 && wrong == 0) ? "true"
                                                                                                    : "false");
        std::fflush(stdout);
    };
    for (int i = 0; i < found; i++)
        if (heur[i].state == CUBLAS_STATUS_SUCCESS) run(heur[i].algo, i, "");
    if (found > 0) {                           // the workaround: the first (default) algorithm, cluster 1x1x1
        cublasLtMatmulAlgo_t a = heur[0].algo;
        uint16_t c1 = CUBLASLT_CLUSTER_SHAPE_1x1x1;
        cublasLtMatmulHeuristicResult_t chk = {};
        if (cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &c1, sizeof c1) ==
                CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulAlgoCheck(lt, op, la, lb, ld, ld, &a, &chk) == CUBLAS_STATUS_SUCCESS &&
            chk.workspaceSize <= wsize)
            run(a, 0, "cluster_1x1x1");
        else
            std::printf("{\"case\": \"%s\", \"n\": \"%s#0w\", \"variant\": \"cluster_1x1x1\", "
                        "\"error\": \"this algorithm does not accept a 1x1x1 cluster\"}\n", name, label);
    }
    std::printf("{\"case\": \"%s_summary\", \"n\": \"%s\", \"algorithms_offered\": %d, \"sms_visible\": %d}\n",
                name, label, found, sms);
    std::fflush(stdout);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulDescDestroy(op);
    cublasLtDestroy(lt);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);
    cudaFree(ws);
}

int main(int argc, char **argv) {
    for (int a = 1; a < argc; a++) {
        std::string arg = argv[a];
        if (arg == "scrub") { scrub(); continue; }
        if (arg.rfind("cpu:", 0) == 0) {          // cpu:<kind>:<n>[:hole], no GPU needed
            std::string rest = arg.substr(4);
            size_t c1 = rest.find(':');
            std::string kind = rest.substr(0, c1);
            int n = c1 == std::string::npos ? 64 : std::atoi(rest.c_str() + c1 + 1);
            cpu_case(kind, n, rest.find(":hole") != std::string::npos);
            continue;
        }
        size_t colon = arg.find(':');
        std::string kind = arg.substr(0, colon);
        if (kind == "lt16" || kind == "ltbf16") {    // lt16:M:K:N
            int M = 1000, K = 2048, N = 16;
            std::sscanf(arg.c_str() + colon + 1, "%d:%d:%d", &M, &K, &N);
            lt_case(kind, M, K, N);
            continue;
        }
        int n = colon == std::string::npos ? 2048 : std::atoi(arg.c_str() + colon + 1);
        if (kind != "hgemm" && kind != "sgemm" && kind != "sgemmtc" && kind != "dgemm" && kind != "ex16" &&
            kind != "exbf16") {
            fail(kind.c_str(), n, "unknown case", -1);
            continue;
        }
        gemm_case(kind, n);
    }
    return 0;
}
