# Nyquist-Support Adapters

## Research hypothesis

Generic rate embeddings showed large initialization sensitivity in our three-seed
study. This motivates a new, unconfirmed architectural hypothesis rather than
selecting a favorable seed. Represent sensor bandwidth at individual FFT patches,
and project unobservable normalized Fourier coordinates out before encoding them.
The classifier and low-rank residuals are learned. There is no reliability gate.

HART represents an even-length FFT as concatenated real and imaginary coordinates
after removing the DC or final component. The support map follows that exact layout
and patch cropping/stride. A Fourier coordinate at physical frequency nu has mask
M(f,nu) = 1[nu <= f/2]. A patch context s is its fraction of observable coordinates.
Time tokens use the global observable fraction f/30. The bounded context is [s,1-s].
The adapter is h + U GELU(D LN(h) + C[s,1-s]), with U initialized to zero.
Entirely unsupported frequency patches are zeroed after adaptation to remove
encoder/adapter bias. Mixed patches are masked coordinate-wise before encoding.
The frozen shared encoder and activity head then operate normally in one forward
pass. No extra teacher pass or alignment term is required.

## Mechanism and scope

For a fixed support mask, P = diag(M) satisfies P^2 = P and ||P||_2 <= 1.
An encoding operator G(Px) has zero derivative with respect to unsupported input
coordinates wherever differentiable. These elementary identities justify the
implementation, not a novel theorem or a guarantee of improved classification.
The rate context lies in [0,1], so metadata below a trained rate does not require
extrapolating an unbounded log-rate feature. This does not bound neural outputs.

Projection acts after HART's existing FFT normalization. Its normalization scales
and the time branch can still depend on the whole input signal. Therefore the
local derivative statement is NOT end-to-end invariance to unavailable physical
frequencies. Masking cannot recover lost high-frequency information or undo
aliasing already folded into the observable band. Naive-decimation controls remain
important. The retained 120-point representation also does not establish reduced
inference FLOPs or sensor energy savings. Device efficiency requires measurement.

## Controls and evaluation

- blind_adapter is the previous unconstrained, rate-blind adapter.
- rate_adapter is the previous global log-rate context.
- support_context uses physical patch context without coordinate projection.
- support_projection uses coordinate projection with context set to zero.
- support_adapter combines projection and context.

Support variants have identical trainable parameter counts and initialization.
Hold training rates, optimizer, window count, and step budget fixed. Report all
seeds and conditions. Start with a bounded comparison, then replicate before any
positive claim. Do not tune the design against the unseen-rate test metrics.
Evaluate both the existing split and a prespecified second split, without choosing
the split for favorable scores. Old reports belong to a different architecture
revision and should not be merged with this comparison.

## Prior work and novelty boundaries

Resampling augmentation already exists in HAR, including Wang et al.,
Sensor Data Augmentation by Resampling for Contrastive Learning in Human Activity
Recognition, https://arxiv.org/abs/2109.02054.
Spectral adapters also exist, including Frame2Freq, a CVPR 2026 video adaptation
method, https://arxiv.org/abs/2602.18977.
Neither resampling nor frequency adapters alone is our novelty claim.
The candidate contribution is integrating measurement-support projection with
parameter-efficient frozen time/FFT encoder adaptation, and testing whether that
constraint improves unseen-rate generalization and initialization stability.
This is a research candidate, not a demonstrated new discovery or acceptance claim.
