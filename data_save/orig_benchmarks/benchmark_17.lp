d3(o1).
d4(o2).
d2(o1).
d1(o2).

d3(X) :- not d0(X), d1(X).
d0(X) :- not d4(X), d1(X).