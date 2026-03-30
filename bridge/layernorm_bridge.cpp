#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../../EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.h"

namespace {

struct Args {
  int party = 0;
  int port = 0;
  int nthreads = 1;
  int ell = 37;
  int scale = 12;
  int dim = 0;
  int array_size = 0;
  std::string address = "127.0.0.1";
  std::string input_path;
  std::string weight_path;
  std::string bias_path;
  std::string output_path;
};

bool parse_int(const char *value, int &out) {
  try {
    out = std::stoi(value);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_args(int argc, char **argv, Args &args) {
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) {
      std::cerr << "Missing value for argument: " << argv[i] << std::endl;
      return false;
    }
    std::string key = argv[i];
    const char *value = argv[i + 1];
    if (key == "--party") {
      if (!parse_int(value, args.party)) return false;
    } else if (key == "--port") {
      if (!parse_int(value, args.port)) return false;
    } else if (key == "--nthreads") {
      if (!parse_int(value, args.nthreads)) return false;
    } else if (key == "--ell") {
      if (!parse_int(value, args.ell)) return false;
    } else if (key == "--scale") {
      if (!parse_int(value, args.scale)) return false;
    } else if (key == "--dim") {
      if (!parse_int(value, args.dim)) return false;
    } else if (key == "--array_size") {
      if (!parse_int(value, args.array_size)) return false;
    } else if (key == "--address") {
      args.address = value;
    } else if (key == "--input") {
      args.input_path = value;
    } else if (key == "--weight") {
      args.weight_path = value;
    } else if (key == "--bias") {
      args.bias_path = value;
    } else if (key == "--output") {
      args.output_path = value;
    } else {
      std::cerr << "Unknown argument: " << key << std::endl;
      return false;
    }
  }
  if ((args.party != 1 && args.party != 2) || args.port <= 0 || args.dim <= 0 || args.array_size <= 0 ||
      args.input_path.empty() || args.weight_path.empty() || args.bias_path.empty() || args.output_path.empty()) {
    return false;
  }
  return true;
}

bool read_u64_file(const std::string &path, std::size_t size, std::vector<uint64_t> &buf) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin) {
    std::cerr << "Failed to open input file: " << path << std::endl;
    return false;
  }
  buf.resize(size);
  fin.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(size * sizeof(uint64_t)));
  if (!fin || fin.gcount() != static_cast<std::streamsize>(size * sizeof(uint64_t))) {
    std::cerr << "Input bytes mismatch. expected=" << (size * sizeof(uint64_t))
              << " got=" << fin.gcount() << std::endl;
    return false;
  }
  return true;
}

bool write_u64_file(const std::string &path, const std::vector<uint64_t> &buf) {
  std::ofstream fout(path, std::ios::binary | std::ios::trunc);
  if (!fout) {
    std::cerr << "Failed to open output file: " << path << std::endl;
    return false;
  }
  fout.write(reinterpret_cast<const char *>(buf.data()),
             static_cast<std::streamsize>(buf.size() * sizeof(uint64_t)));
  return static_cast<bool>(fout);
}

}  // namespace

int main(int argc, char **argv) {
  Args args;
  if (!parse_args(argc, argv, args)) {
    std::cerr << "Usage: layernorm_bridge --party <1|2> --port <int> --address <ip> "
                 "--nthreads <int> --ell <int> --scale <int> --dim <int> --array_size <int> "
                 "--input <path> --weight <path> --bias <path> --output <path>"
              << std::endl;
    return 2;
  }

  const std::size_t total = static_cast<std::size_t>(args.dim) * static_cast<std::size_t>(args.array_size);
  std::cout << "[layernorm_bridge] source=he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.cpp"
            << " function=NonLinear::layer_norm(int, uint64_t*, uint64_t*, uint64_t*, uint64_t*, int, int, int, int)"
            << " party=" << args.party << " port=" << args.port << " nthreads=" << args.nthreads
            << " ell=" << args.ell << " s=" << args.scale << " dim=" << args.dim
            << " array_size=" << args.array_size << std::endl;

  std::vector<uint64_t> in, w, b;
  if (!read_u64_file(args.input_path, total, in)) return 3;
  if (!read_u64_file(args.weight_path, total, w)) return 4;
  if (!read_u64_file(args.bias_path, total, b)) return 5;
  std::vector<uint64_t> out(total, 0ULL);

  NonLinear nl(args.party, args.address, args.port);
  nl.layer_norm(
      args.nthreads,
      in.data(),
      out.data(),
      w.data(),
      b.data(),
      args.dim,
      args.array_size,
      args.ell,
      args.scale);

  if (!write_u64_file(args.output_path, out)) return 6;
  std::cout << "[layernorm_bridge] done party=" << args.party << " wrote=" << args.output_path << std::endl;
  return 0;
}
