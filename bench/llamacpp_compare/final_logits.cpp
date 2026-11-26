// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 inkling-4090-engine contributors
//
// final_logits <gguf> <manifest> <out dir> <single|batch>
//
// llama.cpp side of `astral dump-final-logits`: raw logits after the first L tokens of each
// manifest sequence (lines `name<TAB>L1,L2,...<TAB>ids...`), written as little-endian f32 over
// the model vocabulary to `<name>_<L>.<mode>.f32`.
//   single: one token per llama_decode (like llama-perplexity -ub 1); all lengths in one pass.
//   batch:  the L tokens in one llama_decode with n_ubatch 512 (like -ub 512); cache cleared
//           before each length.
// Build against a llama.cpp tree (see README.md):
//   g++ -O2 -std=c++17 -I$LLAMA/include -I$LLAMA/ggml/include final_logits.cpp \
//       -L$LLAMA/build/bin -lllama -Wl,-rpath,$LLAMA/build/bin -o final_logits
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct Job {
    std::string name;
    std::vector<int> lens;
    std::vector<llama_token> ids;
};

static bool save(const std::string & path, const float * v, int n) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(v, sizeof(float), n, f) == (size_t) n;
    return std::fclose(f) == 0 && ok;
}

int main(int argc, char ** argv) {
    if (argc != 5 || (std::string(argv[4]) != "single" && std::string(argv[4]) != "batch")) {
        std::fprintf(stderr, "usage: %s <gguf> <manifest> <out dir> <single|batch>\n", argv[0]);
        return 2;
    }
    const std::string out = argv[3], mode = argv[4];
    std::vector<Job> jobs;
    int cap = 0;
    std::ifstream mf(argv[2]);
    for (std::string line; std::getline(mf, line);) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        size_t a = line.find('\t'), b = a == std::string::npos ? a : line.find('\t', a + 1);
        if (b == std::string::npos) { std::fprintf(stderr, "bad manifest line: %s\n", line.c_str()); return 2; }
        Job j;
        j.name = line.substr(0, a);
        std::stringstream ls(line.substr(a + 1, b - a - 1));
        for (std::string t; std::getline(ls, t, ',');) j.lens.push_back(std::stoi(t));
        std::stringstream is(line.substr(b + 1));
        for (long t; is >> t;) j.ids.push_back((llama_token) t);
        std::sort(j.lens.begin(), j.lens.end());
        j.lens.erase(std::unique(j.lens.begin(), j.lens.end()), j.lens.end());
        if (j.lens.empty() || j.lens.front() < 1 || j.lens.back() > (int) j.ids.size()) {
            std::fprintf(stderr, "%s: lengths must be in 1..%zu\n", j.name.c_str(), j.ids.size());
            return 2;
        }
        cap = std::max(cap, j.lens.back());
        jobs.push_back(std::move(j));
    }
    if (jobs.empty()) { std::fprintf(stderr, "no sequences in %s\n", argv[2]); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = cap + 16;
    cp.n_batch = cap + 16;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "failed to create the context\n"); return 1; }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch batch = llama_batch_init(cap + 16, 0, 1);
    auto fill = [&](const llama_token * t, int n, int pos0) {
        batch.n_tokens = n;
        for (int i = 0; i < n; i++) {
            batch.token[i] = t[i];
            batch.pos[i] = pos0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i == n - 1;
        }
    };
    for (const Job & j : jobs) {
        llama_memory_clear(llama_get_memory(ctx), true);
        size_t next = 0;
        if (mode == "single") {
            for (int i = 0; i < j.lens.back(); i++) {
                fill(&j.ids[i], 1, i);
                if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "%s: decode failed at %d\n", j.name.c_str(), i); return 1; }
                if (j.lens[next] == i + 1) {
                    if (!save(out + "/" + j.name + "_" + std::to_string(i + 1) + ".single.f32", llama_get_logits_ith(ctx, -1), nv)) return 1;
                    next++;
                }
            }
        } else {
            for (int len : j.lens) {
                llama_memory_clear(llama_get_memory(ctx), true);
                fill(j.ids.data(), len, 0);
                if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "%s: decode failed at length %d\n", j.name.c_str(), len); return 1; }
                if (!save(out + "/" + j.name + "_" + std::to_string(len) + ".batch.f32", llama_get_logits_ith(ctx, -1), nv)) return 1;
            }
        }
        std::printf("%s: %zu lengths up to %d\n", j.name.c_str(), j.lens.size(), j.lens.back());
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
