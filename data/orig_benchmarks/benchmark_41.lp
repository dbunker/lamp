d2(o0).
d0(o1).
d3(o1).
d4(o1).

d3(X) :- d4(X), not d1(X).
d1(X) :- not d4(X), d2(X).