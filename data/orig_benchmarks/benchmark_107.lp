d3(o3).
d3(o0).
d3(o1).
d1(o3).
d4(o0).

d2(X) :- not d0(X), d1(X), not d4(X).
d0(X) :- not d3(X), d1(X), not d4(X).
d4(X) :- not d2(X), not d1(X), d3(X).