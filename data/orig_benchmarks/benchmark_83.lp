d2(o3).
d0(o2).
d1(o0).
d4(o1).

d3(X) :- not d2(X), not d0(X), d4(X).
d1(X) :- not d0(X), d2(X), not d4(X).