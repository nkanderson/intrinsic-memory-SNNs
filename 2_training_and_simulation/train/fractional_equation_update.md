## Goal
Align the fractional-order LIF discretization with the explicit-leak form used by spires (C), while assuming:
- $V_{rest}=0$
- bias folded into $I_n$
- snnTorch reset set to zero (if matching spires), or reset-by-subtraction if preferred

---

## GL approximation (starting point)
$$
D^{\alpha} V(t_n) \approx \frac{1}{\Delta t^{\alpha}} \sum_{k=0}^{H-1} (-1)^k \binom{\alpha}{k} V(t_n - k\Delta t)
$$

Define $g_k = (-1)^k \binom{\alpha}{k}$ and $C = 1/\Delta t^{\alpha}$.

---

## Explicit-leak discrete equation (spires-style)
Choose explicit leak at the previous time step:
$$
D^{\alpha} V(t_n) = -\lambda V(t_{n-1}) + I(t_n)
$$

Substitute GL:
$$
C\sum_{k=0}^{H-1} g_k V(t_n - k\Delta t) = -\lambda V(t_{n-1}) + I(t_n)
$$

Split the $k=0$ term and solve for $V(t_n)$:
$$
V_n = \Delta t^{\alpha}\left(-\lambda V_{n-1} + I_n\right) - \sum_{k=1}^{H-1} g_k V_{n-k}
$$

If $\Delta t = 1$:
$$
V_n = -\lambda V_{n-1} + I_n - \sum_{k=1}^{H-1} g_k V_{n-k}
$$

---

## Lambda defaults / mapping
- spires example default: $\tau_m = 20 \Rightarrow \lambda = 1/\tau_m = 0.05$.
- If we want to approximate a snnTorch-style leak of $\beta=0.9$ using the *implicit* form, the common mapping is:
  $$
  \lambda = \frac{1-\beta}{\beta} \approx 0.111
  $$

  For the *explicit* form, a rough mapping is $\lambda \approx 1-\beta$ (i.e., $0.1$), but this is an approximation and differs from the implicit form.

---

## Implementation steps (summary)
1. **Equation derivation (this doc):** use explicit leak with $V_{n-1}$ as above.
2. **Update Python equation:** change `_base_state_function()` in [2_training_and_simulation/train/fractional_lif.py](2_training_and_simulation/train/fractional_lif.py) to use the explicit-leak form:
  - Use `self.mem` as $V_{n-1}$ in the leak term.
  - Apply $dt^{\alpha}$ scaling to the RHS (or $C^{-1}$).
  - Keep history term as $\sum g_k V_{n-k}$.
3. **Update SV equation:** modify the datapath in [common/sv/neurons/fractional_lif.sv](common/sv/neurons/fractional_lif.sv) to compute:
  $$
  V_n = dt^{\alpha}(-\lambda V_{n-1} + I_n) - \text{history}
  $$
4. **Reset semantics:** if matching spires, use hard reset to zero. If retaining snnTorch default, keep reset-by-subtraction and document the divergence.
5. **Constants/tools:** update any coefficient/constant generation scripts if $dt$ or $\lambda$ representation changes.
6. **Tests:** update or add tests in [2_training_and_simulation/train/test_fractional_lif.py](2_training_and_simulation/train/test_fractional_lif.py) to validate explicit-leak behavior.
