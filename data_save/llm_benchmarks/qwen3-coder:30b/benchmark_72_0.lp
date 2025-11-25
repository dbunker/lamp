d2(o0).
d1(o0).
d2(o1).
d0(o0).

d1(X) :- d2(X), not d0(X).
d3(X) :- d2(X).