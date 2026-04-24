#include <chrono>
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
  std::size_t size = 0;
  std::string address = "127.0.0.1";
  std::string input_path;
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

bool parse_size(const char *value, std::size_t &out) {
  try {
    out = static_cast<std::size_t>(std::stoull(value));
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
    } else if (key == "--size") {
      if (!parse_size(value, args.size)) return false;
    } else if (key == "--address") {
      args.address = value;
    } else if (key == "--input") {
      args.input_path = value;
    } else if (key == "--output") {
      args.output_path = value;
    } else {
      std::cerr << "Unknown argument: " << key << std::endl;
      return false;
    }
  }
  if ((args.party != 1 && args.party != 2) || args.port <= 0 || args.size == 0 ||
      args.input_path.empty() || args.output_path.empty()) {
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
    std::cerr << "Usage: gelu_bridge --party <1|2> --port <int> --address <ip> "
                 "--nthreads <int> --ell <int> --scale <int> --size <int> "
                 "--input <path> --output <path>"
              << std::endl;
    return 2;
  }

  std::cout << "[gelu_bridge] source=he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.cpp"
            << " function=NonLinear::gelu(int, uint64_t*, uint64_t*, int, int, int)"
            << " party=" << args.party << " port=" << args.port << " nthreads=" << args.nthreads
            << " ell=" << args.ell << " s=" << args.scale << " size=" << args.size << std::endl;

  std::vector<uint64_t> in;
  if (!read_u64_file(args.input_path, args.size, in)) return 3;

  std::vector<uint64_t> out(args.size, 0ULL);
  NonLinear nl(args.party, args.address, args.port);
  auto __mpc_t0 = std::chrono::steady_clock::now();
  nl.gelu(args.nthreads, in.data(), out.data(), static_cast<int>(args.size), args.ell, args.scale);
  auto __mpc_t1 = std::chrono::steady_clock::now();
  uint64_t __comm_bytes = 0;
  uint64_t __comm_rounds = 0;
  for (int __i = 0; __i < args.nthreads; __i++) {
    __comm_bytes += nl.iopackArr[__i]->get_comm();
    __comm_rounds += nl.iopackArr[__i]->get_rounds();
  }
  double __elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(__mpc_t1 - __mpc_t0).count() / 1000.0;
  std::cout << "[mpc_stats] party=" << args.party
            << " elapsed_ms=" << __elapsed_ms
            << " comm_bytes=" << __comm_bytes
            << " comm_rounds=" << __comm_rounds << std::endl;

  if (!write_u64_file(args.output_path, out)) return 4;
  std::cout << "[gelu_bridge] done party=" << args.party << " wrote=" << args.output_path << std::endl;
  return 0;
}
