import sys

import graphviz
import matplotlib
import numpy as np
import sklearn
import torch
from sklearn.datasets import make_moons

print("Python      ", sys.version.split()[0])
print("NumPy       ", np.__version__)
print("PyTorch     ", torch.__version__, "| GPU available:", torch.cuda.is_available())
print("matplotlib  ", matplotlib.__version__)
print("scikit-learn", sklearn.__version__)

# Autograd check: derivative of x^3 at x = 2 should be 12
x = torch.tensor(2.0, requires_grad=True)
(x**3).backward()
print("d/dx x^3 at 2 =", x.grad.item())

# Dataset check (you'll train on this later)
X, y = make_moons(n_samples=100, noise=0.1)
print("make_moons  ", X.shape)

# Graphviz check: draws a tiny graph to notes/graphviz_test.png
g = graphviz.Digraph()
g.edge("a", "b")
g.render("notes/graphviz_test", format="png", cleanup=True)
print("Graphviz     OK, see notes/graphviz_test.png")