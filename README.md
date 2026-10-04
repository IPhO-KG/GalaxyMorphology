# GalaxyMorphology

An interpretable computer-vision pipeline for **galaxy morphology classification** and **apparent spiral handedness (CW/CCW)** in deep-field astronomical images.

The project grew from simple image-enhancement experiments (Gaussian smoothing, residuals, unsharp masking, bilateral denoising, log stretching, and gradients) into a batch pipeline that:

1. detects source candidates in a deep-field image;
2. crops each object from the **native-resolution original**;
3. measures morphology-related features;
4. assigns an interpretable morphology candidate;
5. runs a log-polar/Fourier spiral analysis where appropriate;
6. stress-tests CW/CCW under changes in center, annulus, smoothing, noise, and mirroring;
7. exports a CSV catalog and summary plots.

> **Research status:** experimental / research prototype. Morphology scores are heuristic evidence scores, **not calibrated probabilities**. CW/CCW refers to apparent winding on the 2-D image, not the physical 3-D rotation direction of the galaxy.

## Current morphology classes

- Spiral Galaxy
- Elliptical Galaxy
- Edge-on Disk
- Irregular Galaxy
- Merger / Disturbed
- UNCERTAIN
- UNRESOLVED
- NO CLEAR SOURCE

## Method overview

The current pipeline combines two interpretable branches.

### 1. Morphology branch

For each detected source, the code estimates quantities such as:

- source center;
- sky/background and noise;
- axis ratio \(b/a\);
- concentration;
- 180° asymmetry;
- clumpiness;
- number of bright components;
- flux radii such as \(r_{50}\) and \(r_{90}\);
- an SNR proxy.

These are combined into heuristic scores for the morphology classes above.

### 2. Spiral-winding branch

The source is transformed from Cartesian coordinates to a log-polar representation. A logarithmic spiral

\[
r = r_0 e^{b\theta}
\]

becomes approximately linear in \((\theta, \ln r)\). The code then compares Fourier/log-spiral templates for modes \(m=2,3,4\) and positive/negative slopes.

The signed winding score is

\[
S = \frac{P_{\rm CW}-P_{\rm CCW}}{P_{\rm CW}+P_{\rm CCW}}.
\]

A sign alone is **not** accepted as a final result. The pipeline also checks:

- center perturbations;
- multiple radial annuli;
- multiple smoothing scales;
- added-noise stability;
- spiral-vs-straight response;
- explained residual structure;
- phase coherence;
- axis ratio;
- mirror consistency.

If these checks fail, the output stays `UNCERTAIN`.

## Pilot batch result

In the current 120-object pilot catalog:

- 118 objects passed the basic statistics filter;
- 87 received a final `UNCERTAIN` morphology label;
- 11 were classified as Edge-on Disk;
- 9 as Spiral Galaxy;
- 6 as Irregular Galaxy;
- 4 as Merger / Disturbed;
- 1 as Elliptical Galaxy.

No source passed every strict CW/CCW reliability criterion in this pilot. Among the 16 objects that reached the robust winding stage, the diagnostic candidates were:

- 7 CW
- 9 CCW

These numbers are **diagnostics, not an astrophysical asymmetry claim**.

## Repository layout

```text
GalaxyMorphology/
├── README.md
├── requirements.txt
├── .gitignore
├── CITATION.cff
├── docs/
│   ├── METHODOLOGY.md
│   └── RESULTS.md
├── src/
│   └── README.md
├── notebooks/
│   └── README.md
├── data/
│   └── README.md
└── results/
    └── README.md
```

The next code upload should place the reusable analyzer in `src/` and exploratory/batch notebooks in `notebooks/`.

## Quick environment setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

For Jupyter:

```bash
jupyter lab
```

## Recommended data workflow

For exploratory testing, use high-resolution lossless public JWST PNG/TIFF imagery. For scientifically stronger analysis, prefer calibrated survey products (e.g. FITS) and preserve native pixel values.

Do **not** commit very large raw astronomical images to Git. Keep only small examples in the repository and document the source of external datasets.

## Roadmap

- [ ] Move the current hybrid analyzer into `src/`
- [ ] Add the automatic batch-cropping notebook to `notebooks/`
- [ ] Add star/artifact rejection before morphology classification
- [ ] Add deblending for overlapping sources
- [ ] Validate thresholds on labeled galaxies
- [ ] Compare against Galaxy Zoo / expert labels
- [ ] Add calibrated FITS support to the batch workflow
- [ ] Add tests for mirror invariance and synthetic spirals
- [ ] Add a reproducible example dataset
- [ ] Package the pipeline as a small Python module / CLI

## Reproducibility notes

The pipeline is intentionally conservative. `UNCERTAIN` is a valid scientific result: a source is not forced into a morphology or winding class when the available image structure does not support it.

## Author

**Emir Anaraly uulu**  
Independent Researcher, Bishkek, Kyrgyz Republic

## Citation

If you use this repository in academic work, please cite the repository metadata in `CITATION.cff`. A manuscript describing the development from image enhancement to automated morphology and winding analysis is in preparation.
