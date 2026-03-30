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
  int n = 0;
  int dim1 = 0;
  int dim2 = 0;
  int dim3 = 0;
  std::string address = "127.0.0.1";
  std::string input_a_path;
  std::string input_b_path;
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
    } else if (key == "--dim1") {
      if (!parse_int(value, args.dim1)) return false;
    } else if (key == "--dim2") {
      if (!parse_int(value, args.dim2)) return false;
    } else if (key == "--dim3") {
      if (!parse_int(value, args.dim3)) return false;
    } else if (key == "--address") {
      args.address = value;
    } else if (key == "--input_a") {
      args.input_a_path = value;
    } else if (key == "--input_b") {
      args.input_b_path = value;
    } else if (key == "--output") {
      args.output_path = value;
    } else {
      return false;
    }
  }
  return (args.party == 1 || args.party == 2) && args.port > 0 && args.n > 0 && args.dim1 > 0 && args.dim2 > 0 &&
         args.dim3 > 0 && !args.input_a_path.empty() && !args.input_b_path.empty() && !args.output_path.empty();
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
    std::cerr << "Usage: attn_v_matmul_mpc_bridge --party <1|2> --port <int> --address <ip> "
                 "--nthreads <int> --ell <int> --scale <int> --n <int> "
                 "--dim1 <int> --dim2 <int> --dim3 <int> "
                 "--input_a <path> --input_b <path> --output <path>"
              << std::endl;
    return 2;
  }

  const std::size_t input_a_size = static_cast<std::size_t>(args.n) * args.dim1 * args.dim2;
  const std::size_t input_b_size = static_cast<std::size_t>(args.n) * args.dim2 * args.dim3;
  const std::size_t output_size = static_cast<std::size_t>(args.n) * args.dim1 * args.dim3;

  std::cout << "[attn_v_matmul_mpc_bridge] source=he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.cpp"
            << " function=NonLinear::n_matrix_mul_iron(int, uint64_t*, uint64_t*, uint64_t*, int, int, int, int, int, int, int, int, int, int)"
            << " party=" << args.party << " port=" << args.port << " nthreads=" << args.nthreads
            << " ell=" << args.ell << " s=" << args.scale << " n=" << args.n << " dim1=" << args.dim1
            << " dim2=" << args.dim2 << " dim3=" << args.dim3 << std::endl;

  std::vector<uint64_t> input_a, input_b, output(output_size, 0ULL);
  if (!read_u64_file(args.input_a_path, input_a_size, input_a)) return 3;
  if (!read_u64_file(args.input_b_path, input_b_size, input_b)) return 4;

  NonLinear nl(args.party, args.address, args.port);
  nl.n_matrix_mul_iron(args.nthreads, input_a.data(), input_b.data(), output.data(), args.n, args.dim1, args.dim2,
                       args.dim3, args.ell, args.ell, args.ell, args.scale, args.scale, args.scale);

  if (!write_u64_file(args.output_path, output)) return 5;
  std::cout << "[attn_v_matmul_mpc_bridge] done party=" << args.party << " wrote=" << args.output_path << std::endl;
  return 0;
}

