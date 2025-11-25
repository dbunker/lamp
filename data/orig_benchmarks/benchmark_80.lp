d2(o2).
d4(o0).
d4(o2).
d1(o3).

d3(X) :- d2(X), not d4(X).
d4(X) :- not d2(X), d1(X).