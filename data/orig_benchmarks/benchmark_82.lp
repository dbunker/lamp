d0(o3).
d4(o1).
d2(o2).
d1(o2).

d3(X) :- not d2(X), d0(X), d4(X).
d0(X) :- not d3(X), d1(X), d2(X).