d3(o3).
d4(o4).
d1(o0).
d0(o2).

d4(X) :- d0(X), not d1(X).
d0(X) :- not d3(X), d1(X).