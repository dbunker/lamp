d3(o1).
d0(o1).
d1(o0).
d1(o1).
d2(o2).

d0(X) :- d1(X), d3(X), d2(X).
d2(X) :- d1(X), d0(X), d3(X).
d3(X) :- d1(X), d2(X), d0(X).