d3(o2).
d0(o2).
d3(o1).
d2(o1).

d2(X) :- d3(X).
d0(X) :- not d2(X), d3(X).