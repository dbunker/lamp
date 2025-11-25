d1(o4).
d4(o3).
d1(o3).
d1(o1).
d0(o4).

d2(X) :- d1(X), d0(X).
d3(X) :- d1(X), d2(X).
d1(X) :- d2(X), d3(X).