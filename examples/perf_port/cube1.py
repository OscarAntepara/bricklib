from st.expr import Index, ConstRef
from st.grid import Grid

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

# Declare grid
input = Grid("bIn", 3)
output = Grid("bOut", 3)
# Symmetries of 27pt stencil imply 4 constants, permuted +/-
# 0 0 0 - 1 of these
a0 = ConstRef("MPI_C0")
# 0 0 1 - 6 of these
a1 = ConstRef("MPI_C1")
# 0 1 1 - 12 of these
a3 = ConstRef("MPI_C3")
# 1 1 1 - 8 of these
a6 = ConstRef("MPI_C6")

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
       a3 * input(i + 1, j + 1, k) + \
       a3 * input(i - 1, j + 1, k) + \
       a3 * input(i + 1, j - 1, k) + \
       a3 * input(i - 1, j - 1, k) + \
       a3 * input(i + 1, j, k + 1) + \
       a3 * input(i - 1, j, k + 1) + \
       a3 * input(i + 1, j, k - 1) + \
       a3 * input(i - 1, j, k - 1) + \
       a3 * input(i, j + 1, k + 1) + \
       a3 * input(i, j - 1, k + 1) + \
       a3 * input(i, j + 1, k - 1) + \
       a3 * input(i, j - 1, k - 1) + \
       a6 * input(i + 1, j + 1, k + 1) + \
       a6 * input(i - 1, j + 1, k + 1) + \
       a6 * input(i + 1, j - 1, k + 1) + \
       a6 * input(i - 1, j - 1, k + 1) + \
       a6 * input(i + 1, j + 1, k - 1) + \
       a6 * input(i - 1, j + 1, k - 1) + \
       a6 * input(i + 1, j - 1, k - 1) + \
       a6 * input(i - 1, j - 1, k - 1) 
output(i, j, k).assign(calc)

STENCIL = [output]
