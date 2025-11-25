d1(o3).
d3(o1).
d3(o3).
d2(o3).
d2(o0).

d0(X) :- d1(X), d3(X), d2(X).
d3(X) :- d1(X), d2(X), d0(X).