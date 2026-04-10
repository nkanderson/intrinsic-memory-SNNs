#include "Vlif.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  Vlif *lif = new Vlif;

  Verilated::traceEverOn(true);
  VerilatedVcdC *tfp = new VerilatedVcdC;
  lif->trace(tfp, 99);
  tfp->open("wave.vcd");

  // Initialize signals
  lif->clk = 0;
  lif->rst = 1;
  lif->current = 0;

  int cycle = 0;

  // Reset the system
  for (int i = 0; i < 4; i++) {
    lif->clk = !lif->clk;
    lif->eval();
    tfp->dump(cycle * 10);
    cycle++;
  }
  lif->rst = 0;

  // Test with small current
  lif->current = 50;
  for (int i = 0; i < 4; i++) {
    lif->clk = !lif->clk;
    lif->eval();
    tfp->dump(cycle * 10);
    cycle++;
  }

  // Test with larger current
  lif->current = 100;
  for (int i = 0; i < 8; i++) {
    lif->clk = !lif->clk;
    lif->eval();
    tfp->dump(cycle * 10);
    cycle++;
  }

  // Test with current equal to threshold
  lif->current = 200;
  for (int i = 0; i < 8; i++) {
    lif->clk = !lif->clk;
    lif->eval();
    tfp->dump(cycle * 10);
    cycle++;
  }

  // Test with large current
  lif->current = 250;
  for (int i = 0; i < 8; i++) {
    lif->clk = !lif->clk;
    lif->eval();
    tfp->dump(cycle * 10);
    cycle++;
  }

  // Remove current
  lif->current = 0;
  for (int i = 0; i < 20; i++) {
    lif->clk = !lif->clk;
    lif->eval();
    tfp->dump(cycle * 10);
    cycle++;

    // Print state every clock edge
    if (lif->clk) {
      // FIXME: This works with lif->lif__DOT__membrane_potential
      // depending on version of verilator. Need to determine if there's a
      // better way to access this value which will not be version-specific
      // printf("Cycle %d: current=%d, membrane_potential=%d, spike=%d\n",
      //        cycle / 2, lif->current, lif->membrane_potential, lif->spike);
      printf("Cycle %d: current=%d, spike=%d\n", cycle / 2, lif->current,
             lif->spike);
    }
  }

  tfp->close();
  delete lif;
  return 0;
}
