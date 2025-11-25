d1(o3).
d1(o0).
d0(o2).
d2(o2).
d0(o4).

d3(X) :- d1(X), d2(X).
d0(X) :- d2(X), d3(X).
d3(X) :- d2(X), d0(X).