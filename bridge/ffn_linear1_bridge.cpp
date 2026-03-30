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
  int nthreads = 2;
  int ell = 37;
  int scale = 12;
  int n = 0;   // number of independent 1xH matmuls (B*S)
  int h = 0;   // input dim
  int i = 0;   // output dim
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
  for (int idx = 1; idx < argc; idx += 2) {
    if (idx + 1 >= argc) return false;
    std::string key = argv[idx];
    const char *value = argv[idx + 1];
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
    } else if (key == "--n") {
      if (!parse_int(value, args.n)) return false;
    } else if (key == "--h") {
      if (!parse_int(value, args.h)) return false;
    } else if (key == "--i") {
      if (!parse_int(value, args.i)) return false;
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
      return false;
    }
  }
  return (args.party == 1 || args.party == 2) && args.port > 0 && args.n > 0 && args.h > 0 && args.i > 0 &&
         !args.input_path.empty() && !args.weight_path.empty() && !args.bias_path.empty() && !args.output_path.empty();
}

bool read_u64_file(const std::string &path, std::size_t size, std::vector<uint64_t> &buf) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin) return false;
  buf.resize(size);
  fin.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(size * sizeof(uint64_t)));
  return fin && fin.gcount() == static_cast<std::streamsize>(size * sizeof(uint64_t));
}

bool write_u64_file(const std::string &path, const std::vector<uint64_t> &buf) {
  std::ofstream fout(path, std::ios::binary | std::ios::trunc);
  if (!fout) return false;
  fout.write(reinterpret_cast<const char *>(buf.data()),
             static_cast<std::streamsize>(buf.size() * sizeof(uint64_t)));
  return static_cast<bool>(fout);
}

}  // namespace

int main(int argc, char **argv) {
  Args args;
  if (!parse_args(argc, argv, args)) {
    std::cerr << "Usage: ffn_linear1_bridge --party <1|2> --port <int> --address <ip> "
                 "--nthreads <int> --ell <int> --scale <int> --n <int> --h <int> --i <int> "
                 "--input <path> --weight <path> --bias <path> --output <path>"
              << std::endl;
    return 2;
  }

  const int dim1 = 1;
  const std::size_t input_size = static_cast<std::size_t>(args.n) * args.h;
  const std::size_t weight_size = static_cast<std::size_t>(args.n) * args.h * args.i;
  const std::size_t bias_size = static_cast<std::size_t>(args.n) * args.i;
  const std::size_t output_size = static_cast<std::size_t>(args.n) * args.i;
  const uint64_t mask = (args.ell == 64) ? ~0ULL : ((1ULL << args.ell) - 1);

  std::cout << "[ffn_linear1_bridge] source=he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.cpp"
            << " function=NonLinear::n_matrix_mul_iron(int, uint64_t*, uint64_t*, uint64_t*, int, int, int, int, int, int, int, int, int, int)"
            << " party=" << args.party << " port=" << args.port << " nthreads=" << args.nthreads
            << " ell=" << args.ell << " s=" << args.scale << " n=" << args.n << " h=" << args.h
            << " i=" << args.i << std::endl;

  std::vector<uint64_t> input, weight, bias, output(output_size, 0ULL);
  if (!read_u64_file(args.input_path, input_size, input)) return 3;
  if (!read_u64_file(args.weight_path, weight_size, weight)) return 4;
  if (!read_u64_file(args.bias_path, bias_size, bias)) return 5;

  NonLinear nl(args.party, args.address, args.port);
  nl.n_matrix_mul_iron(
      args.nthreads, input.data(), weight.data(), output.data(), args.n, dim1, args.h, args.i, args.ell, args.ell,
      args.ell, args.scale, args.scale, args.scale);

  for (std::size_t idx = 0; idx < output_size; ++idx) {
    output[idx] = (output[idx] + bias[idx]) & mask;
  }

  if (!write_u64_file(args.output_path, output)) return 6;
  std::cout << "[ffn_linear1_bridge] done party=" << args.party << " wrote=" << args.output_path << std::endl;
  return 0;
}
