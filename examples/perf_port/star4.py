from st.expr import Index, ConstRef
from st.grid import Grid

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

# Declare grid
input = Grid("bIn", 3)
output = Grid("bOut", 3)
# Symmetries of 25pt stencil imply 5 constants
# 0 0 0 - 1 of these
a0 = ConstRef("MPI_A0")
# 1 1 1 - 6 of these
a1 = ConstRef("MPI_A1")
# 2 2 2 - 6 of these
a2 = ConstRef("MPI_A2")
# 3 3 3 - 6 of these
a3 = ConstRef("MPI_A3")
# 4 4 4 - 6 of these
a4 = ConstRef("MPI_A4")

# Express computation
# output[i, j, k] is assumed
calc = \
       a0 * input(i, j, k) + \
       a1 * input(i + 1, j, k) + \
       a1 * input(i - 1, j, k) + \
       a1 * input(i, j + 1, k) + \
       a1 * input(i, j - 1, k) + \
       a1 * input(i, j, k + 1) + \
       a1 * input(i, j, k - 1) + \
       a2 * input(i + 2, j, k) + \
       a2 * input(i - 2, j, k) + \
       a2 * input(i, j + 2, k) + \
       a2 * input(i, j - 2, k) + \
       a2 * input(i, j, k + 2) + \
       a2 * input(i, j, k - 2) + \
       a3 * input(i + 3, j, k) + \
       a3 * input(i - 3, j, k) + \
       a3 * input(i, j + 3, k) + \
       a3 * input(i, j - 3, k) + \
       a3 * input(i, j, k + 3) + \
       a3 * input(i, j, k - 3) + \
       a4 * input(i + 4, j, k) + \
       a4 * input(i - 4, j, k) + \
       a4 * input(i, j + 4, k) + \
       a4 * input(i, j - 4, k) + \
       a4 * input(i, j, k + 4) + \
       a4 * input(i, j, k - 4)
output(i, j, k).assign(calc)
STENCIL = [output]