d0(o3).
d1(o1).
d3(o1).
d1(o3).

d2(X) :- d1(X), d3(X).
d1(X) :- d2(X), d0(X).
d0(X) :- d1(X), d2(X).