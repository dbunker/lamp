d2(o1).
d4(o3).
d4(o2).
d1(o2).

d3(X) :- not d2(X), not d0(X), d4(X).
d1(X) :- not d0(X), not d3(X), d4(X).
d0(X) :- not d1(X), d3(X), not d4(X).