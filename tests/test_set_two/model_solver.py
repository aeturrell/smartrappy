from pathlib import Path

with open(Path("equation.tex"), "w") as f:
    f.write(
        "$${\displaystyle {\frac {\partial f_{\alpha }}{\partial t}}+\mathbf {v} _{\alpha }\cdot {\frac {\partial f_{\alpha }}{\partial \mathbf {x} }}+{\frac {q_{\alpha }\mathbf {E} }{m_{\alpha }}}\cdot {\frac {\partial f_{\alpha }}{\partial \mathbf {v} }}=0,}$$"
    )
f.close()
