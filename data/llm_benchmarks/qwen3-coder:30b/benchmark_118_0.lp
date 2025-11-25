d1(o0).
d4(o1).
d3(o1).
d4(o3).

d0(X) :- not d2(X), d1(X).
d2(X) :- d3(X), not d4(X).