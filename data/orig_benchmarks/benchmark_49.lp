d2(o1).
d1(o4).
d2(o2).
d3(o1).

d0(X) :- d1(X), d2(X).
d0(X) :- d2(X), d3(X).
d0(X) :- d1(X), d2(X).