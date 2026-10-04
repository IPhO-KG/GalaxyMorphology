# Data

Large raw astronomical images should not be committed directly to this repository.

Recommended structure on a local machine:

```text
data/
├── raw/          # original JWST / survey products
├── crops/        # native-resolution source crops
└── labels/       # future validation labels
```

For reproducibility, document for each dataset:

- archive / source URL;
- target or field name;
- instrument;
- filters;
- product type;
- original dimensions;
- any preprocessing applied.

For scientific analysis, calibrated FITS products are preferable to display-oriented JPEG composites.
