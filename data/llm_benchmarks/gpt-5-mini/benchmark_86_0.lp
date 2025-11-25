d2(o3).
d4(o2).
d3(o1).
d1(o1).

d0(X) :- not d3(X), d1(X).
d1(X) :- d2(X).