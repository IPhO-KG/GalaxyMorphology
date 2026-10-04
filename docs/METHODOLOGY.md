# Methodology

## Scope

GalaxyMorphology is an interpretable image-analysis baseline for studying visible galaxy structure in deep-field imagery. It is **not yet a calibrated astrophysical classifier** and it is not a neural-network model.

The pipeline has three conceptual stages:

1. source detection and native-resolution cropping;
2. general morphology measurement and scoring;
3. spiral-structure and apparent handedness analysis.

## 1. Source detection and cropping

A detection copy of the full field may be resized for computational speed. Source candidates are detected from a background-subtracted, smoothed luminance map using a robust noise estimate and connected components.

Crucially, the final object crops are taken from the **native-resolution original image**, not from the resized detection copy.

The detector can still return:

- stars;
- diffraction features;
- gravitational arcs;
- blends;
- image artifacts.

For that reason, source detection should not be interpreted as galaxy identification.

## 2. Measured morphology features

The current implementation measures interpretable quantities including:

### Axis ratio

The light distribution is summarized by a weighted second-moment covariance matrix. If its eigenvalues are \(\lambda_{\min}\) and \(\lambda_{\max}\), the approximate projected axis ratio is

\[
q = \frac{b}{a} =
\sqrt{\frac{\lambda_{\min}}{\lambda_{\max}}}.
\]

Values near unity correspond to a rounder projected source; smaller values correspond to stronger elongation.

### Flux radii and concentration

The code estimates radii containing specified fractions of the detected source light, including \(r_{20}\), \(r_{50}\), \(r_{80}\), and \(r_{90}\).

A concentration proxy is constructed from inner and outer flux radii.

### 180-degree asymmetry

The source is compared with a version rotated by 180 degrees around the estimated center. A background correction is included so that noise alone does not dominate the residual.

### Clumpiness

A smoothed version of the source is subtracted from the original and the positive high-frequency residual is used as a proxy for small-scale structure.

### Bright components

Local maxima above a source-dependent threshold provide a rough count of bright components. This is useful for detecting disturbed or multi-component systems, but should not by itself be treated as evidence for a merger.

## 3. Morphology scoring

The current classes are:

- Spiral Galaxy
- Elliptical Galaxy
- Edge-on Disk
- Irregular Galaxy
- Merger / Disturbed

The classifier combines measured features with spiral evidence to form **heuristic scores**. These scores are intentionally not described as probabilities.

The pipeline can reject a forced classification and return:

- `UNCERTAIN`
- `UNRESOLVED`
- `NO CLEAR SOURCE`

The exact thresholds must be calibrated on labeled data before population-level claims are made.

## 4. Log-polar spiral analysis

A logarithmic spiral can be written as

\[
r = r_0 e^{b\theta}.
\]

Taking a logarithm gives

\[
\ln r = \ln r_0 + b\theta,
\]

so logarithmic spiral structure becomes approximately linear in \((\theta, \ln r)\).

The code samples the galaxy in log-polar coordinates, subtracts the axisymmetric radial component, and analyzes the residual.

## 5. Fourier/log-spiral templates

The residual is compared with spiral templates for angular modes

\[
m = 2, 3, 4.
\]

Positive and negative log-spiral slopes encode opposite apparent winding senses. The signed score is

\[
S = \frac{P_{\rm CW}-P_{\rm CCW}}
{P_{\rm CW}+P_{\rm CCW}}.
\]

A positive sign and a negative sign are mapped to opposite apparent winding labels in the image coordinate convention.

**Important:** this is apparent 2-D winding, not a measurement of the physical 3-D rotation vector.

## 6. Reliability checks

A winding sign is accepted only if it remains sufficiently convincing under multiple checks:

- small center shifts;
- different radial annuli;
- different smoothing scales;
- added-noise stress tests;
- spiral-vs-straight comparison;
- explained residual structure;
- radial phase coherence;
- projected axis-ratio filtering;
- mirror consistency.

A mirrored source should reverse the winding score. The mirror test is therefore a useful numerical consistency check, not independent astrophysical evidence.

## 7. Interpretation

The pipeline is designed to prefer `UNCERTAIN` over an unsupported classification. This is intentional.

A future validated version should add:

- star/artifact rejection;
- source deblending;
- calibrated FITS input;
- labeled validation data;
- uncertainty calibration;
- comparison with expert/Galaxy Zoo labels;
- injection tests using synthetic galaxies.
