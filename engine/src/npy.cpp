#include "../include/npy.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace atlas {

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "npy: %s\n", msg.c_str());
    std::exit(1);
}

struct NpyArray {
    std::string descr;            // e.g. "<f4"
    std::vector<int64_t> shape;   // C-order dims
    std::vector<char> data;       // raw array bytes
};

// Extract the value following `key` in the header dict, e.g. key = "'descr':".
// Plain string search — the header is a fixed-form dict literal written by np.save,
// not arbitrary Python.
std::string value_after(const std::string& header, const std::string& key,
                        const std::string& path) {
    const size_t k = header.find(key);
    if (k == std::string::npos) die(path + ": header missing " + key);
    size_t v = k + key.size();
    while (v < header.size() && header[v] == ' ') ++v;
    return header.substr(v);
}

// NPY v1.0 layout: magic, version, uint16 LE header length, ASCII header dict,
// then the raw C-order array bytes.
NpyArray load(const std::string& path) {
    if (!std::ifstream(path).good()) die("cannot open " + path);
    std::ifstream in(path, std::ios::binary);

    char magic[8];
    if (!in.read(magic, 8)) die(path + ": too short for NPY preamble");
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) die(path + ": bad NPY magic");
    if (magic[6] != 1 || magic[7] != 0)
        die(path + ": unsupported NPY version (only 1.0)");

    unsigned char len_bytes[2];
    if (!in.read(reinterpret_cast<char*>(len_bytes), 2)) die(path + ": truncated header length");
    const size_t header_len = static_cast<size_t>(len_bytes[0]) |
                              (static_cast<size_t>(len_bytes[1]) << 8);
    std::string header(header_len, '\0');
    if (!in.read(&header[0], static_cast<std::streamsize>(header_len)))
        die(path + ": truncated header");

    NpyArray arr;

    // 'descr': quoted dtype string. '<f4' and '<i4' only (this machine is LE; assert,
    // don't handle).
    {
        const std::string v = value_after(header, "'descr':", path);
        if (v.empty() || v[0] != '\'') die(path + ": malformed descr");
        const size_t end = v.find('\'', 1);
        if (end == std::string::npos) die(path + ": malformed descr");
        arr.descr = v.substr(1, end - 1);
        if (arr.descr != "<f4" && arr.descr != "<i4")
            die(path + ": unsupported descr '" + arr.descr + "' (only <f4 / <i4)");
    }

    // 'fortran_order': must be False (np.save writes C-order).
    if (value_after(header, "'fortran_order':", path).rfind("False", 0) != 0)
        die(path + ": fortran_order must be False");

    // 'shape': tuple of ints, e.g. (6,) or (6, 32000).
    {
        const std::string v = value_after(header, "'shape':", path);
        if (v.empty() || v[0] != '(') die(path + ": malformed shape");
        const size_t close = v.find(')');
        if (close == std::string::npos) die(path + ": malformed shape");
        size_t pos = 1;
        while (pos < close) {
            while (pos < close && (v[pos] == ' ' || v[pos] == ',')) ++pos;
            if (pos >= close) break;
            char* end = nullptr;
            const long long dim = std::strtoll(v.c_str() + pos, &end, 10);
            if (end == v.c_str() + pos || dim < 0) die(path + ": malformed shape");
            arr.shape.push_back(dim);
            pos = static_cast<size_t>(end - v.c_str());
        }
        if (arr.shape.empty()) die(path + ": scalar (0-d) arrays unsupported");
    }

    int64_t numel = 1;
    for (int64_t d : arr.shape) numel *= d;
    const size_t expected = static_cast<size_t>(numel) * 4;  // both dtypes are 4 bytes
    arr.data.resize(expected);
    if (!in.read(arr.data.data(), static_cast<std::streamsize>(expected)))
        die(path + ": truncated data (expected " + std::to_string(expected) + " bytes)");
    return arr;
}

}  // namespace

Tensor load_npy_f32(const std::string& path) {
    NpyArray arr = load(path);
    if (arr.descr != "<f4") die(path + ": expected <f4, got " + arr.descr);
    Tensor t = Tensor::zeros(arr.shape);
    std::memcpy(t.data, arr.data.data(), arr.data.size());
    return t;
}

std::vector<int> load_npy_i32(const std::string& path) {
    NpyArray arr = load(path);
    if (arr.descr != "<i4") die(path + ": expected <i4, got " + arr.descr);
    std::vector<int> ids(arr.data.size() / 4);
    std::memcpy(ids.data(), arr.data.data(), arr.data.size());
    return ids;
}

}  // namespace atlas
