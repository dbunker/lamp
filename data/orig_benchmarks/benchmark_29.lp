d0(o3).
d3(o3).
d0(o1).
d2(o0).
d2(o3).

d1(X) :- not d2(X), not d3(X), d0(X).
d0(X) :- not d2(X), not d1(X), d3(X).