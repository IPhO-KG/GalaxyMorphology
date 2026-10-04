# Source code

Place reusable Python modules here.

Recommended next refactor:

```text
src/
└── galaxy_morphology/
    ├── __init__.py
    ├── io.py
    ├── detection.py
    ├── morphology.py
    ├── spiral.py
    ├── batch.py
    └── plotting.py
```

The current working notebook/script can be moved here incrementally. Keep numerical analysis functions separate from plotting and notebook UI code so they can be tested independently.
