d3(o0).
d1(o3).
d2(o0).
d1(o0).
d3(o2).

d0(X) :- d1(X), d3(X).
d2(X) :- d0(X), d3(X).
d2(X) :- d1(X), d0(X).