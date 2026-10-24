# Ray tracing and continuous pixel circuit model

The event executable implements `ray_photocurrent_circuit_v2`: CUDA threads query the moving triangle scene and evolve independent analog pixel states. It does not consume image pairs, subtract frames, link ESIM/v2e, or inject events using a prescribed background event rate. Conventional image/depth rendering remains a separate path. The validated CUDA optimization, optical time refinement and native-cell fallback are described in [optimization.md](optimization.md); this preserves the pixel circuit and native camera configuration.

The following primary sources were retrieved from the user's local Zotero library with zotero-mcp. No Internet paper search was used.

| Source | Local Zotero item | What supports this implementation |
|---|---|---|
| Delbrück & Mead, *Analog VLSI adaptive, logarithmic, wide–dynamic-range photoreceptor* (1994) | [CR2R627R](zotero://select/library/items/CR2R627R) | continuous logarithmic transduction, subthreshold voltage scale, intensity-dependent response, active feedback |
| Lichtsteiner, Posch & Delbruck, *A 128×128 120 dB 15 µs latency asynchronous temporal contrast vision sensor* (2008), Sec. II-A/B | [GKBQTLRY](zotero://select/library/items/GKBQTLRY) | active log receptor, source follower, switched-capacitor change amplifier, comparators, self-timed reset and arbitration |
| Brandli et al., *A 240×180 130 dB 3 µs latency global shutter spatiotemporal vision sensor* (2014) | [YVAEM4JN](zotero://select/library/items/YVAEM4JN) | DAVIS pixel architecture; this device is not DAVIS346 |
| Delbruck et al., *Feedback control of event cameras* (2021), Sec. 3.2–3.4 | [XWIH3AM3](zotero://select/library/items/XWIH3AM3) | adjustable thresholds/bandwidth and change amplifier held in reset during refractory period |
| Villa et al., *Image rendering and terrain generation of planetary surfaces using source-available tools* (2023), Eq. 7 and Appendix A | [T86CEEP5](zotero://select/library/items/T86CEEP5) | Hapke direct surface scattering and limiting cases |

These sources support the architecture and qualitative behavior. The equations below are an engineering reduction derived from current balance; they are not claimed to be a verbatim circuit model or transistor-level reproduction of these papers. The earlier behavioral implementation has been replaced.

## Optical and electron path

For every pixel and integration cell, CUDA evaluates camera position/light with cubic Hermite endpoint values and derivatives, and orientation with shortest-arc quaternion interpolation. It traces a fixed stratified grid of rays through the pixel footprint, traverses the BVH, interpolates normals/albedo, tests direct solar visibility, and evaluates Hapke scattering. Misses contribute zero radiance. Fixed spatial samples prevent Monte Carlo rendering jitter from becoming false sensor noise.

Renderer units become band-integrated radiance `L = radiance_scale × rendered_value` in W m⁻² sr⁻¹. In a paraxial lens approximation:

```
E = L × π/(4 f_number²) × lens_transmission × cos⁴(field_angle)
photoelectrons_per_second = E × pixel_pitch² × fill_factor × quantum_efficiency × wavelength/(h c)
dark_electrons_per_second = dark_current_a/q
```

The cos⁴ factor is applied per optical sample. This is an effective monochromatic model; spectral radiance, wavelength-dependent transmission and quantum efficiency would require spectral integration. `radiance_scale` is a required calibration concept: the default is illustrative, not a measured solar radiometric transfer. The trajectory's light intensity must be consistent with this scale. Do not apply a display tone curve, gamma or automatic exposure before this path.

Shot noise is generated as electron counts `N ~ Poisson(rate × Δt)` and enters the receptor as current `qN/Δt`. It is a charge-count **tau-leap approximation**, not exact individual photon arrival simulation. Large counts above 10⁶ use a rounded Gaussian approximation. Dark-current shot noise is included in the same count process. There is no independent ON/OFF noise event generator. Events emerge when the resulting analog voltage crosses a comparator threshold. Optical/current cells must resolve the scene dynamics and receptor response; reducing the cell width tests numerical convergence.

## Continuous receptor and comparators

With `Vscale = k_B T/(q κ)`, receptor voltage is represented by `u = V/Vscale`, up to an arbitrary constant offset. Define the effective stored-charge scale `Ncap = Ceff Vscale/q` and diode-equivalent electron rate `x = exp(u)`. The reduced nonlinear receptor satisfies

```
Ceff Vscale du/dt = q (r - exp(u))
x(t) = r / [1 + (r/x(0) - 1) exp(-r t/Ncap)]
```

For zero input, `x(t) = x(0)/(1+x(0)t/Ncap)`. The implementation evaluates the equivalent inverse-rate expression with `expm1` for numerical stability. The steady voltage is logarithmic in light. Its small-signal time constant is `Ncap/r`: dim pixels respond more slowly without an imposed minimum bandwidth fraction. `Ceff` captures the dominant feedback response and must be fitted to measured latency/bandwidth. It is not necessarily the physical photodiode capacitance. Extra amplifier/source-follower poles, resonance and saturation are not represented by this single-state reduction.

The held change-amplifier reference is compared against `u` with separately fixed ON/OFF thresholds. Reference leakage is `du_ref/dt = -reset_leakage_a/(Creset Vscale)`. Within each constant-current cell, nonlinear threshold roots are found by bisection. A falling receptor plus positive leakage can have an interior minimum; the solver splits there before finding crossings. Thus the comparator is not reduced to endpoint subtraction or a straight line between filtered endpoints.

An event balances the reference at the actual crossing voltage. During `refractory_time`, the reference tracks the receptor and no new events are permitted, while the receptor continues evolving. Optional reset noise perturbs the reference on release with voltage standard deviation `sqrt(k_B T/Creset)`; this kTC approximation is disabled by default because actual switch/feedback noise needs measurement. Fixed threshold mismatch represents fabrication mismatch and is independently drawn for ON/OFF, bounded by `threshold_min`.

Initialization assumes a settled pixel under the initial scene illumination and emits no events. Black pixels initialize with a small positive numerical floor if dark current is disabled. Power-on settling is not simulated.

## Timing, configuration and readout

`scene.time_resolution` is the host trajectory horizon, not timestamp quantization. `physic.circuit_step` bounds the optical/current integration cell. Each CUDA pixel processes multiple cells inside a horizon and can emit multiple events with double-precision crossing times. No intensity images are allocated by the DVS ray caster. The pixel capacity `max_events_per_step` is per host horizon, range 1..64; overflow raises an error instead of silently discarding events.

The GPU trajectory interpolation approximates the CPU trajectory locally; it is not its exact global Bezier curve. Choose horizons small enough to resolve curve changes and keyframes. Midpoint optical sampling can miss occlusions or pulses shorter than `circuit_step`; this is a numerical ray/circuit integrator, not an exact solution of arbitrary discontinuous scenes. Halve both horizon and circuit step for deterministic convergence checks. With stochastic input, compare distributions across step sizes; exact random streams depend on the cell partition.

`config/generic_physical.yaml` supplies neutral illustrative parameters. DAVIS346 and Prophesee profiles preserve their dimensions/intrinsics and threshold examples but share the generic optical/circuit defaults. Their names do not imply calibrated pixel pitch, fill factor, quantum efficiency, capacitance or noise characteristics. Prophesee's specific device model has not been identified.

All optical fractions are in [0,1], lengths are explicitly µm/nm, capacitances F, currents A, temperature K, and times seconds. Thresholds are natural-log equivalent voltage contrast. `ENABLE_PHYSIC_EFFECTS=OFF` disables shot noise, dark current, mismatch, reset hold and leakage; continuous receptor dynamics remain active. The obsolete `ENABLE_HIGH_PASS_FILTER` CMake option has been removed; receptor dynamics are always active. Old YAML keys `dark_current`, `shot_intensity`, `photoreceptor_tau`, `intensity_reference`, `min_bandwidth_fraction`, `leak_rate_hz`, `shot_noise_rate_hz`, `shot_noise_bright_factor`, `hpf_time_constant`, `threshold_noise` are rejected with a migration error; use the dimensioned parameters in the new profiles.

The generic finite FIFO models serialization and additive transport latency, counts dropped requests and drains at the end. It does not reproduce a manufacturer's arbiter/handshake, and bus congestion does not feed back into pixel reset timing. `ENABLE_AER_DELAY` controls this transport model. `readout_bandwidth: 0` bypasses it and preserves comparator crossing timestamps. With transport enabled timestamps are delivery times; simulation output can extend beyond the trajectory end.

Event depth is the geometric ray range averaged over hit spatial samples, re-traced at the **comparator event time**, including for noise events. It is auxiliary scene ground truth, not a DVS depth measurement. Readout latency changes the delivered timestamp without changing this depth; geometry uses meters only if the scene does. Background rays have depth -1.

Legacy binary output remains packed 13 bytes: float32 seconds, int16 x/y, int8 polarity (1 ON, 0 OFF), float32 range. Internal times are double. HDF5 stores int64 microseconds and globally ordered timestamps; `ms_to_idx` is the first index with `t >= 1000*ms`. HDF5 currently does not store depth. Each run copies its YAML and writes model/seed/transport/drop metadata.

## Build, run and validation

```bash
cmake -S . -B build -DBUILD_SENSOR_TESTS=ON
cmake --build build -j8
ctest --test-dir build --output-on-failure
./bin/dvs --dem ../assets/southpole/dem.tif \
  --traj ../assets/southpole/trajectory_cruise.txt \
  --config config/davis346.yaml --event_format hdf5 --output output_davis346
```

Regression tests check analytical nonlinear step crossings, nonuniform event intervals, slower low-light response, both polarities, constant-current partition invariance, reset hold, reference leakage, explicit overflow, electron-count mean and variance, repeatable CUDA shot-current events, FIFO serialization, file formats and Hapke limits. A CUDA test traces a real triangle scene with identical horizon endpoint illumination and an interior pulse: it produces both ON and OFF events. Halving the optical/current cell checks crossing convergence. This specifically detects an implementation that only compares endpoint images.

Historical 64×48 regression (before enforcing native camera profiles): the user DEM crop produced 1056 deterministic moving events and zero static events. Halving and quartering the steps preserved each pixel’s event count and polarity sequence. Nevertheless, four coarse/half-step event times differed by more than 10 µs (maximum 151 µs); half/quarter-step comparison also retained sensitive timestamps. Complex DEM timing is not uniformly converged to microseconds. Detailed statistics and remaining uncertainty are in `../audit/2026-10-03/circuit_v2/review.md`.

Full-scene rendering cost scales with pixels × spatial samples × optical cells × BVH traversal. A DAVIS profile with 16² samples and 1 µs cells can be very expensive. Tests on a small DEM crop do not establish full-resolution throughput. Source files and historical outputs were initially audited in `../audit/2026-10-03/review.md`; subsequent circuit validation is saved in `../audit/2026-10-03/circuit_v2/`.

Hardware validation still requires measured radiometric transfer, latency versus irradiance, ON/OFF mismatch, temperature-dependent dark current/leakage, reset behavior and readout saturation. This is a physically dimensioned reduced model, not a calibrated device or SPICE model. It omits flicker/1-f noise, amplifier thermal noise except optional reset kTC, hot/dead pixels, radiation damage, parasitic photocurrent, proprietary denoising and multiple circuit poles.

The renderer uses one directional source, hard shadows and Hapke's regolith scattering term. It omits terrain-to-terrain indirect illumination and solar-disk integration. Sensor noise cannot replace indirect illumination in permanently shadowed terrain. The cleanup fixes NoData/scale/offset loading, I/F inversion, local camera mounting and binary timestamp units. Planar mesh input requires an axis-aligned common CRS; rotated/sheared rasters and mismatched CRS produce an error and need prior reprojection. Missing material data falls back to configured homogeneous SSA and is reported; this is not a measured albedo recovery. The DEM pipeline remains a local planar heightfield rather than a global curved planetary mesh. Blender metadata intrinsics/normal export were corrected, but an installed Blender runtime is needed to validate the exporter end to end.


## Cleanup regression

`test_render` runs with the complete DAVIS346 and Prophesee configuration files, retaining resolution, intrinsics and 16² spatial samples. It checks image coordinate conventions, repeat rendering, partial-coverage depth, static circuit initialization, I/F-to-SSA round trips and camera mounts. `test_sensor` uses native DAVIS resolution/intrinsics with one spatial sample and controlled circuit settings solely to isolate circuit convergence/noise; it is not a calibrated sensor experiment. `test_raster` checks scaled NoData, mesh validity, alignment boundaries/axis order and binary timestamp/matrix loading. All outputs and the pre-cleanup source snapshot are recorded in `../audit/2026-10-03/cleanup/`.
