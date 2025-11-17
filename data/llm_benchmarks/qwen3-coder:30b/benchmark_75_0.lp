d1(o1).
d2(o2).
d3(o2).
d1(o2).

d3(X) :- d2(X), d1(X).
d2(X) :- d1(X), d3(X).
d1(X) :- d2(X), d3(X).