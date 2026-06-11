// NPY reader tests — Phase 1 test hardening.
//
// Step 4's reader self-test proves the happy path on the committed oracles; this
// target owns the format contract. It writes synthetic NPY v1.0 files byte-by-byte
// (so the expected parse is known by construction, including a non-64-aligned header
// the docs say must not be relied on) and exercises every documented die() with the
// self-subprocess pattern: the exe re-invokes itself with a case name, the child runs
// the death scenario, and the parent CHECKs the child exited non-zero. Blob-free.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../include/npy.h"

static int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// Build an NPY v1.0 byte stream: magic, version, uint16 LE header length, the ASCII
// dict (space-padded, '\n'-terminated so the preamble hits `align`), raw data.
static std::vector<char> make_npy(const std::string& descr, const std::string& shape,
                                  const void* data, size_t data_bytes, size_t align = 64) {
    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " +
                         shape + ", }";
    const size_t preamble = 6 + 2 + 2;  // magic + version + header-length field
    while ((preamble + header.size() + 1) % align != 0) header += ' ';
    header += '\n';

    std::vector<char> out;
    const char magic[8] = {'\x93', 'N', 'U', 'M', 'P', 'Y', '\x01', '\x00'};
    out.insert(out.end(), magic, magic + 8);
    out.push_back(static_cast<char>(header.size() & 0xff));
    out.push_back(static_cast<char>((header.size() >> 8) & 0xff));
    out.insert(out.end(), header.begin(), header.end());
    const char* p = static_cast<const char*>(data);
    out.insert(out.end(), p, p + data_bytes);
    return out;
}

static void write_file(const std::string& path, const std::vector<char>& bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// --- happy paths ----------------------------------------------------------------------

static void test_f32_2d() {
    std::printf("test_npy: <f4 (2, 3) round trip\n");
    const float vals[6] = {1.5f, -2.0f, 3.25f, 4.0f, -5.5f, 6.0f};
    write_file("tmp_npy_f32_2d.npy", make_npy("<f4", "(2, 3)", vals, sizeof(vals)));
    const atlas::Tensor t = atlas::load_npy_f32("tmp_npy_f32_2d.npy");
    CHECK(t.shape == std::vector<int64_t>({2, 3}));
    for (int64_t i = 0; i < 6; ++i) CHECK(t.data[i] == vals[i]);
}

static void test_f32_1d_unaligned_header() {
    // np.save 64-aligns the preamble; the spec says read exactly HEADER_LEN bytes and
    // never rely on the padding width — this file aligns to 16 to prove it.
    std::printf("test_npy: <f4 (4,) with non-64-aligned header\n");
    const float vals[4] = {0.0f, -0.125f, 1e-3f, 12345.0f};
    write_file("tmp_npy_f32_1d.npy", make_npy("<f4", "(4,)", vals, sizeof(vals), 16));
    const atlas::Tensor t = atlas::load_npy_f32("tmp_npy_f32_1d.npy");
    CHECK(t.shape == std::vector<int64_t>({4}));
    for (int64_t i = 0; i < 4; ++i) CHECK(t.data[i] == vals[i]);
}

static void test_i32_1d() {
    std::printf("test_npy: <i4 (5,) round trip\n");
    const int32_t vals[5] = {1, -1, 0, 2147483647, -2147483647};
    write_file("tmp_npy_i32_1d.npy", make_npy("<i4", "(5,)", vals, sizeof(vals)));
    const std::vector<int> ids = atlas::load_npy_i32("tmp_npy_i32_1d.npy");
    CHECK(ids == std::vector<int>({1, -1, 0, 2147483647, -2147483647}));
}

// --- death cases (child mode) ---------------------------------------------------------
// Each writes its malformed file and calls the loader; die() exits 1. Reaching the
// return means the reader accepted bad input — the child exits 0 and the parent's
// CHECK on the exit code fails.

static int run_death_case(const std::string& which) {
    const float one = 1.0f;
    if (which == "bad-magic") {
        std::vector<char> b = make_npy("<f4", "(1,)", &one, 4);
        b[0] = 'X';
        write_file("tmp_npy_bad_magic.npy", b);
        atlas::load_npy_f32("tmp_npy_bad_magic.npy");
    } else if (which == "bad-version") {
        std::vector<char> b = make_npy("<f4", "(1,)", &one, 4);
        b[6] = '\x02';  // NPY v2.0: unsupported by contract
        write_file("tmp_npy_bad_version.npy", b);
        atlas::load_npy_f32("tmp_npy_bad_version.npy");
    } else if (which == "fortran-order") {
        std::vector<char> b = make_npy("<f4", "(1,)", &one, 4);
        std::string s(b.begin(), b.end());
        const size_t at = s.find("False");
        std::memcpy(&b[at], "True ", 5);  // same header length, flipped flag
        write_file("tmp_npy_fortran.npy", b);
        atlas::load_npy_f32("tmp_npy_fortran.npy");
    } else if (which == "bad-descr") {
        const double d = 1.0;
        write_file("tmp_npy_f8.npy", make_npy("<f8", "(1,)", &d, 8));
        atlas::load_npy_f32("tmp_npy_f8.npy");
    } else if (which == "truncated") {
        const float two[2] = {1.0f, 2.0f};
        write_file("tmp_npy_trunc.npy", make_npy("<f4", "(100,)", two, 8));
        atlas::load_npy_f32("tmp_npy_trunc.npy");
    } else if (which == "dtype-mismatch") {
        const int32_t iv = 7;
        write_file("tmp_npy_i4.npy", make_npy("<i4", "(1,)", &iv, 4));
        atlas::load_npy_f32("tmp_npy_i4.npy");  // f32 loader on an i32 file
    } else if (which == "missing-file") {
        atlas::load_npy_f32("tmp_npy_does_not_exist.npy");
    } else {
        std::fprintf(stderr, "unknown death case: %s\n", which.c_str());
        return 2;
    }
    return 0;  // loader accepted bad input — parent will flag this as a failure
}

static void expect_death(const char* exe, const std::string& which) {
    const std::string cmd = "\"" + std::string(exe) + "\" " + which;
    const int rc = std::system(cmd.c_str());
    std::printf("test_npy: death case '%s' -> child exit %d\n", which.c_str(), rc);
    CHECK(rc != 0);
}

int main(int argc, char** argv) {
    if (argc > 1) return run_death_case(argv[1]);  // child mode

    test_f32_2d();
    test_f32_1d_unaligned_header();
    test_i32_1d();

    for (const char* c : {"bad-magic", "bad-version", "fortran-order", "bad-descr",
                          "truncated", "dtype-mismatch", "missing-file"}) {
        expect_death(argv[0], c);
    }

    if (g_failures == 0) {
        std::printf("test_npy: all checks passed\n");
        return 0;
    }
    std::printf("test_npy: %d check(s) FAILED\n", g_failures);
    return 1;
}
