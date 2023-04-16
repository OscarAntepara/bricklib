from st.expr import Index, ConstRef
from st.grid import Grid

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

# Declare grid
input = Grid("bIn", 3)
output = Grid("bOut", 3)
# Symmetries of 7pt stencil imply 2 constants
# 0 0 0 - 1 of these
a0 = ConstRef("MPI_ALPHA")
# 1 1 1 - 6 of these
a1 = ConstRef("MPI_BETA")

# Express computation
# output[i, j, k] is assumed
calc = \
       a0 * input(i, j, k) + \
       a1 * input(i + 1, j, k) + \
       a1 * input(i - 1, j, k) + \
       a1 * input(i, j + 1, k) + \
       a1 * input(i, j - 1, k) + \
       a1 * input(i, j, k + 1) + \
       a1 * input(i, j, k - 1)
output(i, j, k).assign(calc)
STENCIL = [output]