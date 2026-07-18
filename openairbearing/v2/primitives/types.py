"""Array type aliases for the v2 pipeline.

Scientific aliases (AxisScalar, FieldScalar, …) read as the physics;
raw aliases (F64, I32) for generic array code, including ``@njit`` kernels.
"""

import numpy as np
import numpy.typing as npt

# ── Raw dtypes ──────────────────────────────────────────────────────────────
type Bool = npt.NDArray[np.bool_]
type F64 = npt.NDArray[np.float64]
type I32 = npt.NDArray[np.int32]
type I64 = npt.NDArray[np.int64]

# ── Semantic aliases (all resolve to F64 at runtime) ────────────────────────
type AxisScalar = F64
"""(n,) grid node coordinates along one axis."""
type FieldScalar = F64
"""(nx,) or (nx, ny) spatial field values."""
type SampleScalar = F64
"""(nh,) one value per film-thickness sample."""
type StackScalar = F64
"""(nh, nx) or (nh, nx, ny) per-sample field stack."""
