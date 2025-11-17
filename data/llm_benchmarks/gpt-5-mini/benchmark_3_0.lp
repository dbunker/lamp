d3(o2).
d1(o1).
d0(o3).
d2(o1).

d3(X) :- d2(X), not d0(X).
d2(X) :- not d1(X), d3(X).