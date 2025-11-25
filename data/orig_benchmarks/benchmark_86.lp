d2(o3).
d4(o2).
d3(o1).
d1(o1).

d4(X) :- not d1(X), d2(X).
d0(X) :- d1(X), not d3(X).
d1(X) :- d2(X), not d3(X).