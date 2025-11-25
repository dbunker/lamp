d4(o4).
d0(o1).
d2(o1).
d4(o2).
d0(o4).

d1(X) :- d2(X), d0(X), not d3(X).
d3(X) :- not d1(X), d2(X), d0(X).
d1(X) :- d0(X), d3(X), not d4(X).