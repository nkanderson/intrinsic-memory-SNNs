#include "Vtop.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

int main(int argc, char **argv)
{
  Verilated::commandArgs(argc, argv);

  Vtop *top = new Vtop;

  Verilated::traceEverOn(true);
  VerilatedVcdC *tfp = new VerilatedVcdC;
  top->trace(tfp, 99);
  tfp->open("wave.vcd");

  top->clk = 0;
  top->rst = 1;

  for (int i = 0; i < 20; i++)
  {
    if (i == 2)
      top->rst = 0;

    top->clk = !top->clk;
    top->eval();
    tfp->dump(i * 10); // 10 ns per cycle
  }

  tfp->close();
  delete top;
  return 0;
}
