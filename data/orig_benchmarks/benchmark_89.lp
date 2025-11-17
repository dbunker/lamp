d3(o0).
d1(o1).
d4(o0).
d0(o2).

d2(X) :- not d3(X), d0(X).
d2(X) :- not d4(X), d0(X).