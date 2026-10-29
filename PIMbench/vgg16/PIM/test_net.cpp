// Multi-layer ON-CNN test: runs a small CNN (conv / relu / pool stages) through the same
// utilML.h functions the VGG16 benchmark uses and compares EVERY stage with the Python reference.
#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <getopt.h>
#include <iomanip>
#include <chrono>
#include "util.h"
#include "utilML.h"

using Tensor = std::vector<std::vector<std::vector<int>>>;
using Kernel4 = std::vector<Tensor>;

struct Params
{
  char *dramConfigFile = nullptr;
  const char *metaFile = "net_meta.txt";
  const char *inputFile = "net_input.txt";
  const char *weightFile = "net_weights.txt";
  const char *expectedFile = "net_expected.txt";
};

struct Stage
{
  std::string type;     // "conv", "relu", "pool"
  int a = 0, b = 0, c = 0, d = 0, e = 0;
  std::string mode;     // conv: "4d" (true Cout x Cin x Hf x Wf) or "3d" (shared kernel, as the VGG16 benchmark uses)
};

static std::vector<int> readFlat(const char *fn)
{
  std::vector<int> v;
  std::ifstream f(fn);
  if (!f.is_open()) { std::cerr << "Error: cannot open " << fn << std::endl; return v; }
  int x;
  while (f >> x) v.push_back(x);
  return v;
}

int main(int argc, char **argv)
{
  Params p;
  int opt;
  while ((opt = getopt(argc, argv, "c:d:i:w:e:")) >= 0)
  {
    switch (opt)
    {
    case 'c': p.dramConfigFile = optarg; break;
    case 'd': p.metaFile = optarg; break;
    case 'i': p.inputFile = optarg; break;
    case 'w': p.weightFile = optarg; break;
    case 'e': p.expectedFile = optarg; break;
    default: std::cerr << "usage: test_net.out [-c cfg] [-d meta] [-i input] [-w weights] [-e expected]\n"; return 1;
    }
  }

  // ---- network description ---------------------------------------------------------------
  int C = 0, H = 0, W = 0;
  std::vector<Stage> stages;
  {
    std::ifstream m(p.metaFile);
    std::string tok;
    if (!m.is_open()) { std::cerr << "Error: cannot open " << p.metaFile << std::endl; return 1; }
    while (m >> tok)
    {
      Stage s;
      s.type = tok;
      if (tok == "input") m >> C >> H >> W;
      else if (tok == "conv") { m >> s.a >> s.b >> s.c >> s.d >> s.e >> s.mode; stages.push_back(s); } // cout kh kw stride pad mode
      else if (tok == "relu") stages.push_back(s);
      else if (tok == "pool") { m >> s.a >> s.b; stages.push_back(s); }                                   // k stride
      else { std::cerr << "Error: unknown token '" << tok << "' in meta file\n"; return 1; }
    }
  }
  std::vector<int> flatIn = readFlat(p.inputFile), flatW = readFlat(p.weightFile), flatExp = readFlat(p.expectedFile);
  if (C <= 0 || stages.empty() || flatIn.size() != (size_t)C * H * W)
  {
    std::cerr << "Error: input file does not match the meta file\n";
    return 1;
  }

  Tensor cur(C, std::vector<std::vector<int>>(H, std::vector<int>(W)));
  size_t ii = 0;
  for (auto &ch : cur) for (auto &row : ch) for (int &v : row) v = flatIn[ii++];

  if (!createDevice(p.dramConfigFile)) return 1;

  size_t wpos = 0, epos = 0;
  int totalErrors = 0, firstBad = -1;

  for (size_t si = 0; si < stages.size(); ++si)
  {
    const Stage &s = stages[si];
    Tensor out;
    std::string desc = s.type;

    if (s.type == "conv")
    {
      const int cin = cur.size(), cout = s.a, kh = s.b, kw = s.c;
      desc += " " + s.mode + " (cin=" + std::to_string(cin) + " cout=" + std::to_string(cout) + " k=" +
              std::to_string(kh) + "x" + std::to_string(kw) + " stride=" + std::to_string(s.d) + " pad=" + std::to_string(s.e) + ")";
      if (s.mode == "4d")
      {
        if (wpos + (size_t)cout * cin * kh * kw > flatW.size()) { std::cerr << "Error: weight file too short\n"; return 1; }
        Kernel4 k(cout, Tensor(cin, std::vector<std::vector<int>>(kh, std::vector<int>(kw))));
        for (auto &f : k) for (auto &c : f) for (auto &r : c) for (int &v : r) v = flatW[wpos++];
        conv2(cur, k, out, s.d, s.e);
      }
      else
      {
        if (wpos + (size_t)cout * kh * kw > flatW.size()) { std::cerr << "Error: weight file too short\n"; return 1; }
        Tensor k(cout, std::vector<std::vector<int>>(kh, std::vector<int>(kw)));
        for (auto &f : k) for (auto &r : f) for (int &v : r) v = flatW[wpos++];
        conv2(cur, k, out, s.d, s.e);
      }
    }
    else if (s.type == "relu")
    {
      out = cur;
      relu(out);
    }
    else // pool
    {
      desc += " " + std::to_string(s.a) + "x" + std::to_string(s.a) + " stride " + std::to_string(s.b);
      pool(cur, s.a, s.a, s.b, out);
    }

    // compare this stage with the reference
    const size_t n = out.size() * (out.empty() ? 0 : out[0].size()) * (out.empty() || out[0].empty() ? 0 : out[0][0].size());
    if (epos + n > flatExp.size()) { std::cerr << "Error: expected file too short at stage " << si << "\n"; return 1; }
    int errs = 0;
    for (auto &ch : out) for (auto &row : ch) for (int v : row)
    {
      if (v != flatExp[epos])
      {
        if (errs < 3)
          std::cout << "  mismatch stage " << si << " flat index " << (epos) << ": expected " << flatExp[epos] << " got " << v << std::endl;
        ++errs;
      }
      ++epos;
    }
    std::cout << "stage " << si << ": " << desc << " -> [" << out.size() << "][" << out[0].size() << "][" << out[0][0].size() << "]  "
              << (errs ? "MISMATCH x" + std::to_string(errs) : std::string("OK")) << std::endl;
    if (errs && firstBad < 0) firstBad = (int)si;
    totalErrors += errs;
    cur = out;   // keep chaining our own result, exactly like the benchmark does
  }

  if (epos != flatExp.size() || wpos != flatW.size())
  {
    std::cout << "FAILED: file sizes do not match the network description (generator and test disagree)\n";
    return 1;
  }

  if (totalErrors == 0)
    std::cout << "\nSUCCESS: all " << stages.size() << " stages match the reference\n\n";
  else
    std::cout << "\nFAILED: " << totalErrors << " mismatches, first bad stage = " << firstBad << "\n\n";

  pimShowStats();
  return totalErrors == 0 ? 0 : 1;
}