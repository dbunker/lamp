d2(o2).
d1(o2).
d3(o1).
d3(o0).

d0(X) :- not d2(X), d1(X).
d1(X) :- d3(X).