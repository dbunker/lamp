d0(o1).
d3(o2).
d2(o2).
d4(o1).

d1(X) :- not d3(X), d4(X).
d4(X) :- not d0(X), d1(X).