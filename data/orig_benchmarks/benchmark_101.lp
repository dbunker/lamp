d0(o3).
d4(o1).
d0(o1).
d2(o3).
d1(o0).

d0(X) :- not d2(X), d1(X), not d4(X).
d3(X) :- not d2(X), not d1(X), d0(X).