d3(o1).
d2(o1).
d3(o2).
d1(o2).

d0(X) :- d3(X), not d1(X), not d2(X).
d0(X) :- not d3(X), not d1(X), d2(X).
d0(X) :- not d2(X), not d3(X), d1(X).