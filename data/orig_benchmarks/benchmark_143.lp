d0(o3).
d2(o4).
d3(o0).
d1(o0).
d2(o1).

d4(X) :- not d2(X), not d1(X), d3(X).
d2(X) :- not d3(X), d0(X), not d4(X).
d0(X) :- not d2(X), not d1(X), d3(X).