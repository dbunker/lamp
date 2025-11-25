d3(o3).
d0(o1).
d2(o3).
d1(o0).
d1(o1).

d2(X) :- d1(X), d0(X), not d3(X).
d0(X) :- d1(X), d2(X), not d3(X).