d2(o4).
d0(o2).
d1(o3).
d0(o3).

d2(X) :- d0(X), not d1(X).
d3(X) :- not d0(X), d1(X).