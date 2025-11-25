d2(o4).
d1(o4).
d1(o1).
d3(o2).
d0(o4).

d1(X) :- d0(X), not d2(X).
d3(X) :- d0(X), not d1(X).
d3(X) :- not d1(X), d2(X).