d0(o3).
d1(o3).
d3(o3).
d2(o0).
d0(o2).

d1(X) :- d2(X), d3(X), d4(X).
d4(X) :- d1(X), d0(X), d3(X).
d0(X) :- d1(X), d2(X), d4(X).