d3(o3).
d2(o3).
d0(o2).
d1(o3).

d4(X) :- not d3(X), d0(X).
d1(X) :- d0(X), not d2(X).