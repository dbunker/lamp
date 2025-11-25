d4(o0).
d4(o3).
d3(o3).
d1(o0).

d0(X) :- d4(X), not d2(X).
d2(X) :- not d3(X), d1(X).