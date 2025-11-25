d2(o2).
d0(o2).
d4(o2).
d1(o3).

d1(X) :- not d3(X), d0(X), d4(X).
d3(X) :- d2(X), d0(X), not d4(X).
d3(X) :- d1(X), d0(X), not d4(X).