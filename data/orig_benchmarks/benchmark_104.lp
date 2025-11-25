d4(o1).
d3(o3).
d1(o1).
d3(o2).
d2(o2).

d4(X) :- not d1(X), d0(X).
d0(X) :- not d3(X), d4(X).
d1(X) :- d2(X), not d3(X).