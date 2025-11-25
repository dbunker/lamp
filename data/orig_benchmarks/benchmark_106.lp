d0(o1).
d1(o1).
d0(o2).
d4(o2).
d2(o2).

d1(X) :- not d2(X), d3(X), d4(X).
d1(X) :- d2(X), d0(X), not d4(X).
d3(X) :- not d1(X), d2(X), d0(X).