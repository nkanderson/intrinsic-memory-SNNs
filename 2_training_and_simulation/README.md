# Training

## Models

After Optuna studies, five models were chosen for training:
- Smallest LIF: 32hl1, 16hl2 (trial 81 from leaky-1000ep study)
- A larger LIF: 64hl1, 16hl2 (trial 44 from leaky-1000ep study)
- Smallest fractional-order LIF: 16hl1, 4hl2, history length of 32
- Shortest history-length fractional-order LIF: 32hl1, 4hl2, history length of 16 (Original best model came from config with -run2 naming variant. The duplicative files have been cleaned, and the -run2 naming has been removed.)
- Custom slow decay bitshift LIF: 32hl1, 8hl2, history length of 8

In training, the best generalized model was chosen from each.
The bitshift model was quantized to QS3.12, the others were all quantized to QS2.13.

## Steps

1. Training:
```bash
python main.py --config configs/optimized-leaky-32hl1-16hl2.yaml
python main.py --config configs/optimized-leaky-64hl1-16hl2.yaml
python main.py --config configs/fractional-16hl1-4hl2-32hist.yaml
python main.py --config configs/fractional-32hl1-4hl2-16hist.yaml
python main.py --config configs/bitshift-custom_slow_decay-32hl1-8hl2-8hist.yaml
```

2. Generate coefficients .mem file and SV module constants (for fractional and bitshift only):
```bash
python scripts/generate_coefficients.py --lam 0.17601697332161148 --output-dir ../../common/sv/cocotb/tests/weights/fractional-16-4-32 --coeff-frac-bits 16 --history-length 32
python scripts/generate_coefficients.py --lam 0.05263950403873735 --output-dir ../../common/sv/cocotb/tests/weights/fractional-32-4-16 --coeff-frac-bits 16 --history-length 16
python scripts/generate_coefficients.py --lam 0.09550310156350894 --output-dir ../../common/sv/cocotb/tests/weights/bitshift-custom_slow_decay --history-length 8 --constants-only
```

3. Generate quantized model .pth file for validation in software simulation (PyTorch):
```bash
python scripts/manage_weights.py export pytorch models/dqn_64hl1-16hl2-best.pth --bits 16 --frac 13 --signed
python scripts/generate_coefficients.py --lam 0.05263950403873735 --output-dir ../../common/sv/cocotb/tests/weights/fractional-32-4-16 --coeff-frac-bits 16 --history-length 16
```

4. Run software simulation to validate quantized model:
```bash
python main.py --load models/dqn_fractional-64hl1-16hl2-best-quantized-QS2_13.pth --evaluate-only --max-episode-steps 5000
python main.py --load models/dqn_64hl1-16hl2-best-quantized-QS2_13.pth --evaluate-only
python main.py --load models/dqn_bitshift-custom_slow_decay-32hl1-8hl2-8hist-best-generalization-quantized-QS3_12.pth --evaluate-only --config configs/bitshift-custom_slow_decay-32hl1-8hl2-8hist.yaml
```

5. Export quantized weights to .mem files use in hardware simulation (using cocotb):
```bash
python scripts/manage_weights.py export hardware models/dqn_optimized-leaky-32hl1-16hl2-best-generalization.pth --bits 16 --frac 13 --output ../../common/sv/cocotb/tests/weights/lif-32-16
python scripts/manage_weights.py export hardware models/dqn_optimized-leaky-64hl1-16hl2-best-generalization.pth --bits 16 --frac 13 --output ../../common/sv/cocotb/tests/weights/lif-64-16
python scripts/manage_weights.py export hardware models/dqn_fractional-16hl1-4hl2-32hist-best-generalization.pth --bits 16 --frac 13 --output ../../common/sv/cocotb/tests/weights/fractional-16-4-32
python scripts/manage_weights.py export hardware models/dqn_fractional-32hl1-4hl2-16hist-run2-best-generalization.pth --bits 16 --frac 13 --output ../../common/sv/cocotb/tests/weights/fractional-32-4-16
```

## V2 Models

1. Training:
```bash
python main.py --config configs/bitshift-custom_slow_decay-32hl1-8hl2-8hist.yaml
python main.py --config configs/fractional-16hl1-4hl2-32hist.yaml

```
