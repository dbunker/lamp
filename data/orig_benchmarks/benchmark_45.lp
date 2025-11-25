d0(o3).
d0(o1).
d2(o1).
d3(o1).

d2(X) :- d1(X), d0(X), d3(X).
d1(X) :- d2(X), d0(X), d3(X).