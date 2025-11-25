d2(o3).
d4(o2).
d1(o0).
d1(o2).

d3(X) :- not d4(X), not d1(X).
d3(X) :- d4(X).