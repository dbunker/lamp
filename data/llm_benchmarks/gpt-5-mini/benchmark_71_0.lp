d0(o3).
d3(o4).
d1(o0).
d2(o2).
d0(o4).

d0(X) :- not d1(X), d2(X).
d2(X) :- d0(X), not d3(X).