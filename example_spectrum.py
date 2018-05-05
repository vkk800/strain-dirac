"""A quick example that computes and plots the non-superconducting
spectrum in the Tang-Fu potential with two different values of \beta.
"""

from scipy import *
from matplotlib.pyplot import *
import diracp as d
import tools as t


kspace = linspace(-35,35,101)

spec = []
spec2 = []
for kk in kspace:
    print("k is: {}".format(kk))
    
    sol = d.solveDiraci(t.TFpot(20, 30), t.zerod(30), 1e-5, kk, i0=1, i1=2)
    esols = array([ss['E'] for ss in sol])
    spec.append(esols)
    sol = d.solveDiraci(t.TFpot(30, 30), t.zerod(30), 1e-5, kk, i0=1, i1=2)
    esols = array([ss['E'] for ss in sol])
    spec2.append(esols)

spec = asarray(spec)
spec2 = asarray(spec2)

plot(kspace, spec[:,0], 'r-', kspace, spec2[:,0], 'b--',
     kspace, spec[:,1], 'r-', kspace, spec2[:,1], 'b--')

axis([-35, 35, -20, 20])
xlabel(r"$k_y L$")
ylabel(r"$E L / v_F$")
legend([r"$\beta=20$", r"$\beta=30$"])
savefig("example_spectrum.pdf")
show()
