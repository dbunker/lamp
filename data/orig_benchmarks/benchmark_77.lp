d3(o0).
d2(o3).
d1(o2).
d0(o1).

d0(X) :- not d2(X), d1(X).
d3(X) :- d0(X), not d2(X).