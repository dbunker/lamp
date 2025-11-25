d3(o1).
d1(o1).
d0(o3).
d1(o2).

d2(X) :- not d0(X), d3(X).
d2(X) :- d3(X), not d1(X).