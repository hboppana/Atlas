// WeightStore tests — Phase 1 test hardening.
//
// The mmap + manifest loader has only ever been exercised by the 4.4 GB real blob
// (blob-gated). This target writes a synthetic 56-byte blob + manifest, so the
// contract — non-owning contiguous views by name, byte-offset arithmetic, the
// documented die()s on a malformed manifest — is covered on any machine. Death cases
// use the self-subprocess pattern (child re-invocation by case name). Blob-free.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "../include/model.h"

static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// 14 floats, values 0..13: a [2,3] at byte 0, b [4] at 24, c [2,2] at 40 — contiguous,
// like the real converter output (manifest total == blob size).
static void write_fixture(const std::string& bin, const std::string& manifest_text,
                          const std::string& manifest) {
    std::vector<float> blob(14);
    for (size_t i = 0; i < blob.size(); ++i) blob[i] = static_cast<float>(i);
    std::ofstream b(bin, std::ios::binary | std::ios::trunc);
    b.write(reinterpret_cast<const char*>(blob.data()),
            static_cast<std::streamsize>(blob.size() * sizeof(float)));
    std::ofstream m(manifest, std::ios::trunc);
    m << manifest_text;
}

static const char* kGoodManifest = "a 0 2 2 3\nb 24 1 4\nc 40 2 2 2\n";

static void test_views() {
    std::printf("test_weightstore: views over a synthetic 56-byte blob\n");
    write_fixture("tmp_ws_good.bin", kGoodManifest, "tmp_ws_good.txt");
    const auto ws = atlas::WeightStore::load("tmp_ws_good.bin", "tmp_ws_good.txt");

    CHECK(ws.has("a"));
    CHECK(ws.has("c"));
    CHECK(!ws.has("z"));

    const atlas::Tensor& a = ws.get("a");
    CHECK(a.shape == std::vector<int64_t>({2, 3}));
    CHECK(!a.owns);  // a view into the mapping, not a copy
    CHECK(a.at({0, 0}) == 0.0f);
    CHECK(a.at({1, 2}) == 5.0f);

    const atlas::Tensor& b = ws.get("b");
    CHECK(b.shape == std::vector<int64_t>({4}));
    CHECK(b.at({3}) == 9.0f);

    const atlas::Tensor& c = ws.get("c");
    CHECK(c.shape == std::vector<int64_t>({2, 2}));
    CHECK(c.at({1, 1}) == 13.0f);

    // Contiguity of the mapping: the manifest's byte offsets are element offsets into
    // one mmap'd buffer, so consecutive tensors are pointer-adjacent.
    CHECK(b.data == a.data + 6);
    CHECK(c.data == b.data + 4);
}

// --- death cases (child mode) ---------------------------------------------------------

static int run_death_case(const std::string& which) {
    if (which == "missing-name") {
        write_fixture("tmp_ws_dn.bin", kGoodManifest, "tmp_ws_dn.txt");
        const auto ws = atlas::WeightStore::load("tmp_ws_dn.bin", "tmp_ws_dn.txt");
        ws.get("nope");  // dies: names come from model code, absence is a bug
    } else if (which == "misaligned-offset") {
        write_fixture("tmp_ws_mis.bin", "a 2 1 4\n", "tmp_ws_mis.txt");  // 2 % 4 != 0
        atlas::WeightStore::load("tmp_ws_mis.bin", "tmp_ws_mis.txt");
    } else if (which == "past-end") {
        write_fixture("tmp_ws_end.bin", "a 0 2 100 100\n", "tmp_ws_end.txt");
        atlas::WeightStore::load("tmp_ws_end.bin", "tmp_ws_end.txt");
    } else if (which == "empty-manifest") {
        write_fixture("tmp_ws_empty.bin", "", "tmp_ws_empty.txt");
        atlas::WeightStore::load("tmp_ws_empty.bin", "tmp_ws_empty.txt");
    } else if (which == "missing-blob") {
        write_fixture("tmp_ws_mb.bin", kGoodManifest, "tmp_ws_mb.txt");
        atlas::WeightStore::load("tmp_ws_nonexistent.bin", "tmp_ws_mb.txt");
    } else {
        std::fprintf(stderr, "unknown death case: %s\n", which.c_str());
        return 2;
    }
    return 0;  // loader accepted bad input — parent flags this as a failure
}

static void expect_death(const char* exe, const std::string& which) {
    const std::string cmd = "\"" + std::string(exe) + "\" " + which;
    const int rc = std::system(cmd.c_str());
    std::printf("test_weightstore: death case '%s' -> child exit %d\n", which.c_str(), rc);
    CHECK(rc != 0);
}

int main(int argc, char** argv) {
    if (argc > 1) return run_death_case(argv[1]);  // child mode

    test_views();
    for (const char* c : {"missing-name", "misaligned-offset", "past-end",
                          "empty-manifest", "missing-blob"}) {
        expect_death(argv[0], c);
    }

    if (g_failures == 0) {
        std::printf("test_weightstore: all checks passed\n");
        return 0;
    }
    std::printf("test_weightstore: %d check(s) FAILED\n", g_failures);
    return 1;
}
