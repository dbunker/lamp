d2(o0).
d2(o1).
d0(o4).
d1(o3).

d3(X) :- not d2(X), not d1(X), d0(X).
d0(X) :- not d2(X), not d1(X), d3(X).