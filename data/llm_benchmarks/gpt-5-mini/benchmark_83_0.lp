d2(o3).
d0(o2).
d1(o0).
d4(o1).

d1(X) :- not d0(X), d2(X).
d3(X) :- d4(X), not d1(X).