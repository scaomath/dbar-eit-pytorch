# Patch validation

## Files changed
- `dbar.py`
- `torch_dbar/dbar.py`
- `example_circular.py`

## Sanity checks run

### 1. Syntax
```bash
python -m py_compile /mnt/data/dbar.py /mnt/data/torch_dbar/dbar.py /mnt/data/example_circular.py
```
Passed.

### 2. DN-map invariance under rescaling of current patterns
Tested on homogeneous conductivity with adjacent currents of amplitudes `1.0` and `0.5`.

Result:
- `||DN(amp=1.0)|| = 3.8713535566973234`
- `||DN(amp=0.5)|| = 3.8713535566973234`
- relative difference `= 0.0`

This is the expected behavior after normalizing by the actual applied current patterns.

### 3. Reconstruction path still runs
A reduced square-domain Born reconstruction test completed successfully after the patch.

Result:
- DN shapes: `(15, 15)`
- Reconstruction shape: `(1, 16, 16)`
- Reconstructed range: `0.9730599969366942` to `1.029290568753394`

## Scope of the patch
This patch fixes the measurement-to-ND/DN construction so it uses the actual current patterns and becomes invariant to current scaling.

It does **not** implement the missing boundary-integral-equation / CGO stage of the full continuum D-bar pipeline.
