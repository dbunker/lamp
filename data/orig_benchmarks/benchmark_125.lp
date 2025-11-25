d4(o4).
d0(o1).
d2(o3).
d1(o0).

d3(X) :- not d0(X), d2(X), not d4(X).
d2(X) :- not d1(X), d3(X), not d4(X).
d1(X) :- not d0(X), d2(X), not d3(X).