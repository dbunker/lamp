d3(o0).
d3(o1).
d4(o0).
d1(o1).
d2(o2).

d4(X) :- not d1(X), d2(X), d3(X).
d0(X) :- not d2(X), d3(X), d4(X).