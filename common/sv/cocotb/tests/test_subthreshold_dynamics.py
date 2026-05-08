import csv
import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
from pathlib import Path

# Fixed-point / module constants
DATA_WIDTH = 16
MEMBRANE_WIDTH = 24
FRAC_BITS = 13
SCALE = 1 << FRAC_BITS

def wrap_signed(val: int, bits: int) -> int:
    mask = (1 << bits) - 1
    val &= mask
    if val >= (1 << (bits - 1)):
        val -= 1 << bits
    return val

def float_to_fixed_qs2_13(x: float) -> int:
    raw = int(round(x * SCALE))
    return wrap_signed(raw, DATA_WIDTH)

def signal_to_signed(value: int, bits: int) -> int:
    return wrap_signed(int(value), bits)

async def reset_dut(dut):
    dut.reset.value = 1
    dut.clear.value = 0
    dut.enable.value = 0
    dut.current.value = 0
    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)

async def step_dut(dut, current_signed: int):
    dut.current.value = current_signed & ((1 << DATA_WIDTH) - 1)
    dut.enable.value = 1
    await RisingEdge(dut.clk)
    dut.enable.value = 0

    if int(dut.output_valid.value) == 0:
        max_wait_cycles = 1024
        for _ in range(max_wait_cycles):
            await RisingEdge(dut.clk)
            if int(dut.output_valid.value) == 1:
                break
        else:
            raise AssertionError("Timed out waiting for output_valid")

    spike = int(dut.spike_out.value)
    membrane = signal_to_signed(int(dut.membrane_out.value), MEMBRANE_WIDTH)
    return spike, membrane

@cocotb.test()
async def test_subthreshold_dynamics(dut):
    """
    Simulates charge and discharge phases to capture subthreshold dynamics.
    Dumps the membrane potential trace to a CSV file.
    """
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    steps_charge = 50
    steps_discharge = 150
    total_steps = steps_charge + steps_discharge
    
    input_current = 0.1
    input_qs2_13 = float_to_fixed_qs2_13(input_current)
    
    mem_trace = []
    
    for t in range(total_steps):
        # Apply current for charge phase, 0 for discharge
        i_val = input_qs2_13 if t < steps_charge else 0
        
        spike, mem = await step_dut(dut, i_val)
        
        # Convert fixed point back to float
        mem_float = mem / float(1 << FRAC_BITS)
        mem_trace.append(mem_float)
        
        # Ensure we didn't spike (it's supposed to be subthreshold)
        assert spike == 0, f"Unexpected spike at t={t}"

    # Figure out if this is fractional or bitshift based on the module name or params
    # We can pass an environment variable to set the name, or try to read a parameter
    model_name = os.environ.get("TEST_MODEL_NAME", "unknown_model")
    
    # Save to CSV
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    csv_path = results_dir / f"subthreshold_{model_name}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "membrane_potential"])
        for t, m in enumerate(mem_trace):
            writer.writerow([t, m])
            
    cocotb.log.info(f"Saved subthreshold trace to {csv_path}")
