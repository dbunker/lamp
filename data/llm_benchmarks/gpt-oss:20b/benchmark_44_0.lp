d1(o3).
d3(o0).
d3(o1).
d0(o0).

d2(X) :- d0(X), not d1(X).
d1(X) :- d3(X), not d0(X).