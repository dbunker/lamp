d0(o3).
d4(o3).
d3(o3).
d3(o1).
d2(o0).

d4(X) :- d1(X), d2(X), d0(X).
d1(X) :- d0(X), d3(X), d4(X).