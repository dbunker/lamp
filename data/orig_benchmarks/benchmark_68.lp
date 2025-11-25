d0(o1).
d1(o1).
d0(o2).
d1(o2).
d0(o4).

d3(X) :- not d0(X), d2(X).
d2(X) :- d1(X), not d3(X).
d0(X) :- d2(X), not d3(X).