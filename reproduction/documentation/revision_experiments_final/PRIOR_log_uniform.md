# Prior Specification: `log_uniform`

All probability/sensitivity statements are conditional on this declared prior.

| Parameter | Range | Family | Notes |
|---|---:|---|---|
| β_D | [0.05, 1.0] | log_uniform | defection temptation |
| β_κ | [0.5, 3.0] | uniform | conflict / deterrence |
| β_α | [0.2, 1.5] | uniform | coordination / signal value |
| β_Ω | [0.5, 2.5] | uniform | oversight / membership value |
| β_ℓ | [0.8, 3.0] | log_uniform | removal / outside option |
| N | [2, 10] | uniform | sampled as discrete-uniform over integers (rounded after sampling on [1.5, 10.5]) |
| δ | [0.1, 0.95] | uniform | discount factor prior |

Uniform priors are the maximum-entropy choice given bounds; log-uniform priors stress-test order-of-magnitude uncertainty.
