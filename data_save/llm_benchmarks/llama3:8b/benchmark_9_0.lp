d2(o0).
d3(o2).
d1(o2).
d2(o1).

d0(X) :- not d2(X), d1(X).
d3(o2) :- not d2(o2).