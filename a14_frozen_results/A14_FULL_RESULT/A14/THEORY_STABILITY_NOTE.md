# A14 analytical stability note: exact quadratic proximal reconciliation

For a fixed calibrated operator \(A\), measurement \(y\), and proximal step \(\gamma>0\), define

\[
P_{A,y}(z)=\operatorname{prox}_{\gamma g_{A,y}}(z)
=(I+\gamma A^\top A)^{-1}(z+\gamma A^\top y),
\qquad
g_{A,y}(x)=\tfrac12\|Ax-y\|_2^2.
\]

The IC-PRO-S reconciliation with a fixed relaxation \(\eta\in[0,1]\) is

\[
T_{\eta,A,y}(z)=\Pi_+\!\left[(1-\eta)z+\eta P_{A,y}(z)\right].
\]

## 1. Nonexpansiveness in the learned proposal

The proximal map of a proper closed convex function is firmly nonexpansive, hence
\(\|P(z_1)-P(z_2)\|_2\le \|z_1-z_2\|_2\).
A convex combination of the identity with a nonexpansive map is nonexpansive for
\(\eta\in[0,1]\), and the nonnegative projection \(\Pi_+\) is also nonexpansive. Therefore

\[
\|T_{\eta,A,y}(z_1)-T_{\eta,A,y}(z_2)\|_2\le \|z_1-z_2\|_2.
\]

## 2. Measurement perturbation bound

Because \(H_A=I+\gamma A^\top A\succeq I\), \(\|H_A^{-1}\|_2\le1\). For a measurement perturbation
\(\delta y\),

\[
P_{A,y+\delta y}(z)-P_{A,y}(z)
=\gamma H_A^{-1}A^\top\delta y,
\]

so

\[
\|P_{A,y+\delta y}(z)-P_{A,y}(z)\|_2
\le \gamma\|A\|_2\|\delta y\|_2.
\]

For fixed \(\eta\), the corresponding under-relaxed and projected output perturbation is bounded by the same
quantity multiplied by \(\eta\).

## 3. Operator perturbation bound

Let \(B=A+E\), \(p_A=P_{A,y}(z)\), and \(H_B=I+\gamma B^\top B\). Then

\[
H_B(P_{B,y}(z)-p_A)
=\gamma\left[(A^\top A-B^\top B)p_A+(B^\top-A^\top)y\right].
\]

Using \(\|H_B^{-1}\|_2\le1\) and
\(\|A^\top A-B^\top B\|_2\le(\|A\|_2+\|B\|_2)\|E\|_2\),

\[
\|P_{B,y}(z)-P_{A,y}(z)\|_2
\le \gamma\|E\|_2
\left[(\|A\|_2+\|B\|_2)\|p_A\|_2+\|y\|_2\right].
\]

For a fixed \(\eta\), the under-relaxed reconciliation scales this envelope by \(\eta\).
If the learned reliability changes from \(\eta_A\) to \(\eta_B\), before projection,

\[
\|\tilde T_B-\tilde T_A\|_2
\le \eta_B\|P_B-P_A\|_2
+|\eta_B-\eta_A|\,\|P_A-z\|_2.
\]

Thus the learned gate has a clear interpretation: it controls the amount of operator/measurement sensitivity introduced
by the exact data-fidelity reconciliation while retaining an explicit path toward the proximal point.

A14 is the matched empirical perturbation test for this analytical envelope. It does not alter or retrain IC-PRO-S.
